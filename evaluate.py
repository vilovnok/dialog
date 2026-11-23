import argparse
import json
import logging
import threading

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

from classifier.llm import GigaChatClient
from classifier.decision_model import GigaChatDecisionModel
from classifier.decision_tree import DecisionTree
from classifier.schemas import InputData
from classifier.context_builder import build_context
from classifier.categories import (
    FILE_CATEGORY,
    QUESTIONS_CONSECUTIVE_CATEGORY,
    CORRECT_CATEGORY,
)


INPUT_PATH = Path(
    "/home/work/22850885@sigma.sbrf.ru/project/workplaces/data/tagme_balanced_small_dataset.csv"
)

OUTPUT_PATH = Path("data/evaluation_results.csv")
METRICS_PATH = Path("data/evaluation_metrics.json")

INPUT_COLUMN = "INPUT:input"
GOLDEN_COLUMN = "GOLDEN:agent"

LOG_EVERY = 10

worker_local = threading.local()


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# --------------------------------------------------
# WORKER INITIALIZATION
# --------------------------------------------------

def get_worker_decision_tree() -> DecisionTree:
    """Возвращает DecisionTree текущего потока."""

    if not hasattr(worker_local, "decision_tree"):

        llm_client = GigaChatClient()

        decision_model = GigaChatDecisionModel(
            client=llm_client,
        )

        worker_local.decision_tree = DecisionTree(
            model=decision_model,
        )

        logger.info(
            "Создан DecisionTree для потока %s",
            threading.current_thread().name,
        )

    return worker_local.decision_tree


# --------------------------------------------------
# INPUT PROCESSING
# --------------------------------------------------

def get_target_message_id(
    data: dict[str, Any],
) -> str | None:
    """Извлекает ID целевого сообщения."""

    for message in data.get("answer", []):
        if str(message.get("Ключ для разметки")) == "1":
            return message.get("ID запроса")

    return None


def process_dialog(
    raw_data: dict[str, Any],
    decision_tree: DecisionTree,
) -> dict[str, Any]:
    """Обрабатывает диалог через DecisionTree."""

    input_data = InputData.model_validate(raw_data)

    context = build_context(
        input_data,
        messages_before=None,
        messages_after=None,
    )

    result = decision_tree.run(context)

    return result.model_dump(mode="json")


# --------------------------------------------------
# DECISION ANALYSIS
# --------------------------------------------------

def get_step(
    decisions: dict[str, Any],
    step_name: str,
) -> dict[str, Any]:
    """Безопасно возвращает результат отдельного этапа."""

    return decisions.get(step_name) or {}


def get_final_source(
    workflow_result: dict[str, Any],
) -> str:
    """
    Определяет этап, который сформировал итоговую категорию.

    Приоритет соответствует логике DecisionTree:
    spam -> file -> questions_consecutive -> correct -> category.
    """

    decisions = workflow_result.get("decisions", {})
    final_label = workflow_result.get("label")

    if workflow_result.get("spam"):
        return "spam"

    file_step = get_step(decisions, "file")

    if (
        file_step.get("status") == "completed"
        and final_label == FILE_CATEGORY
    ):
        return "file"

    questions_step = get_step(
        decisions,
        "questions_consecutive",
    )

    if (
        questions_step.get("status") == "completed"
        and final_label == QUESTIONS_CONSECUTIVE_CATEGORY
    ):
        return "questions_consecutive"

    correct_step = get_step(decisions, "correct")

    if (
        correct_step.get("status") == "completed"
        and final_label == CORRECT_CATEGORY
    ):
        return "correct"

    return "category"


def get_final_reason(
    workflow_result: dict[str, Any],
    final_source: str,
) -> str | None:
    """Возвращает причину выбора итоговой категории."""

    decisions = workflow_result.get("decisions", {})

    step = get_step(decisions, final_source)

    return step.get("reason")


def get_category_type(
    workflow_result: dict[str, Any],
) -> str | None:
    """Извлекает тип результата классификации категории."""

    decisions = workflow_result.get("decisions", {})
    category_step = get_step(decisions, "category")

    return category_step.get("category_type")


# --------------------------------------------------
# ROW EVALUATION
# --------------------------------------------------

def evaluate_row(
    row: pd.Series,
    decision_tree: DecisionTree,
    row_index: int,
) -> dict[str, Any]:
    """Обрабатывает одну строку датасета."""

    golden = row.get(GOLDEN_COLUMN)
    filename = row.get("FILENAME")

    result = {
        "row_index": row_index,
        "message_id": None,
        "filename": filename,

        # Эталон и итоговый прогноз
        "golden": golden,
        "prediction": None,
        "is_correct": None,

        # Источник итогового решения
        "final_source": None,
        "final_reason": None,

        # Spam
        "spam": None,
        "spam_status": None,
        "spam_label": None,
        "spam_reason": None,

        # File
        "file_status": None,
        "file_label": None,
        "file_reason": None,

        # Questions consecutive
        "questions_consecutive_status": None,
        "questions_consecutive_label": None,
        "questions_consecutive_reason": None,

        # Correct
        "correct_status": None,
        "correct_label": None,
        "correct_reason": None,

        # Category
        "category_status": None,
        "category_type": None,
        "category_label": None,
        "category_reason": None,

        # Общий статус
        "workflow_status": "failed",
        "error": None,
    }

    try:
        raw_data = json.loads(row[INPUT_COLUMN])

        result["message_id"] = get_target_message_id(
            raw_data
        )

        workflow_result = process_dialog(
            raw_data=raw_data,
            decision_tree=decision_tree,
        )

        decisions = workflow_result.get("decisions", {})

        spam_step = get_step(decisions, "spam")
        file_step = get_step(decisions, "file")

        questions_step = get_step(
            decisions,
            "questions_consecutive",
        )

        correct_step = get_step(decisions, "correct")
        category_step = get_step(decisions, "category")

        # Определяем этап, который сформировал итоговую категорию.
        final_source = get_final_source(workflow_result)

        # Сохраняем результаты всех этапов.
        result.update({

            # Итоговый результат
            "prediction": workflow_result.get("label"),
            "spam": workflow_result.get("spam"),

            "final_source": final_source,
            "final_reason": get_final_reason(
                workflow_result,
                final_source,
            ),

            # Spam
            "spam_status": spam_step.get("status"),
            "spam_label": spam_step.get("label"),
            "spam_reason": spam_step.get("reason"),

            # File
            "file_status": file_step.get("status"),
            "file_label": file_step.get("label"),
            "file_reason": file_step.get("reason"),

            # Questions consecutive
            "questions_consecutive_status": (
                questions_step.get("status")
            ),
            "questions_consecutive_label": (
                questions_step.get("label")
            ),
            "questions_consecutive_reason": (
                questions_step.get("reason")
            ),

            # Correct
            "correct_status": correct_step.get("status"),
            "correct_label": correct_step.get("label"),
            "correct_reason": correct_step.get("reason"),

            # Category
            "category_status": category_step.get("status"),
            "category_type": get_category_type(
                workflow_result
            ),
            "category_label": category_step.get("label"),
            "category_reason": category_step.get("reason"),

            # Workflow
            "workflow_status": (
                workflow_result.get("workflow", {}).get("status")
            ),
        })

        if pd.notna(golden) and result["prediction"] is not None:
            result["is_correct"] = (
                str(result["prediction"]).strip()
                == str(golden).strip()
            )

    except Exception as exc:
        result["error"] = str(exc)

        logger.exception(
            "Ошибка обработки строки %s",
            row_index,
        )

    return result


def evaluate_row_parallel(
    row: pd.Series,
    row_index: int,
) -> dict[str, Any]:
    """Обрабатывает строку в текущем потоке."""

    decision_tree = get_worker_decision_tree()

    return evaluate_row(
        row=row,
        decision_tree=decision_tree,
        row_index=row_index,
    )


# --------------------------------------------------
# METRICS
# --------------------------------------------------

def calculate_metrics(
    results_df: pd.DataFrame,
) -> dict[str, Any]:
    """Рассчитывает основные метрики классификации."""

    total = len(results_df)

    successful = results_df[
        results_df["workflow_status"] == "completed"
    ]

    predicted = successful[
        successful["prediction"].notna()
    ]

    comparable = predicted[
        predicted["golden"].notna()
    ]

    correct_count = int(
        comparable["is_correct"].sum()
    )

    accuracy = (
        correct_count / len(comparable)
        if len(comparable) > 0
        else None
    )

    # Распределение источников итоговых решений.
    final_source_counts = (
        results_df["final_source"]
        .value_counts(dropna=False)
        .to_dict()
    )

    # Распределение предсказанных категорий.
    prediction_counts = (
        results_df["prediction"]
        .value_counts(dropna=False)
        .to_dict()
    )

    return {
        "total_rows": total,
        "completed_rows": len(successful),
        "failed_rows": int(
            (results_df["workflow_status"] == "failed").sum()
        ),
        "spam_rows": int(
            results_df["spam"].fillna(False).sum()
        ),
        "predicted_rows": len(predicted),
        "comparable_rows": len(comparable),
        "correct_predictions": correct_count,
        "incorrect_predictions": (
            len(comparable) - correct_count
        ),
        "accuracy": accuracy,
        "coverage": (
            len(predicted) / total
            if total > 0
            else None
        ),
        "final_source_distribution": final_source_counts,
        "prediction_distribution": prediction_counts,
    }


# --------------------------------------------------
# ARGUMENTS
# --------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Параллельная оценка диалогов через LLM."
    )

    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=1,
        help="Количество параллельных воркеров (по умолчанию: 1).",
    )

    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers должен быть не меньше 1.")

    return args


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main() -> None:

    args = parse_args()
    workers = args.workers

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    METRICS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 1. Читаем CSV

    df = pd.read_csv(
        INPUT_PATH,
        encoding="utf-8-sig",
        dtype=str,
    )

    if INPUT_COLUMN not in df.columns:
        raise ValueError(
            f"Не найден столбец {INPUT_COLUMN}"
        )

    if GOLDEN_COLUMN not in df.columns:
        raise ValueError(
            f"Не найден столбец {GOLDEN_COLUMN}"
        )

    total_rows = len(df)

    logger.info("Загружено строк: %s", total_rows)
    logger.info("Количество воркеров: %s", workers)

    # 2. Подготавливаем результаты.

    results = [None] * total_rows

    # 3. Параллельная обработка.

    with ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix="llm-worker",
    ) as executor:

        futures = {
            executor.submit(
                evaluate_row_parallel,
                row,
                position,
            ): position

            for position, (_, row) in enumerate(
                df.iterrows()
            )
        }

        completed = 0

        for future in as_completed(futures):

            position = futures[future]

            try:
                results[position] = future.result()

            except Exception as exc:
                logger.exception(
                    "Необработанная ошибка строки %s",
                    position,
                )

                results[position] = {
                    "row_index": position,
                    "workflow_status": "failed",
                    "error": str(exc),
                }

            completed += 1

            if (
                completed % LOG_EVERY == 0
                or completed == total_rows
            ):

                logger.info(
                    "Обработано %s/%s",
                    completed,
                    total_rows,
                )

                completed_results = [
                    item
                    for item in results
                    if item is not None
                ]

                pd.DataFrame(
                    completed_results
                ).to_csv(
                    OUTPUT_PATH,
                    index=False,
                    encoding="utf-8-sig",
                )

    # 4. Итоговое сохранение.

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # 5. Метрики.

    metrics = calculate_metrics(results_df)

    METRICS_PATH.write_text(
        json.dumps(
            metrics,
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    logger.info("Обработка завершена.")
    logger.info("Accuracy: %s", metrics["accuracy"])
    logger.info("Workers: %s", workers)
    logger.info(
        "Источники решений: %s",
        metrics["final_source_distribution"],
    )
    logger.info("Результаты: %s", OUTPUT_PATH)
    logger.info("Метрики: %s", METRICS_PATH)


if __name__ == "__main__":
    main()