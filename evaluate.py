
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from classifier.llm import GigaChatClient
from classifier.decision_model import GigaChatDecisionModel
from classifier.decision_tree import DecisionTree
from classifier.schemas import InputData
from classifier.context_builder import build_context


INPUT_PATH = Path(
    "/Users/richardgurtsiev/Desktop/projects/vibe_code/workspaces/data/tagme_balanced_small_dataset.csv"
)

OUTPUT_PATH = Path("data/evaluation_results.csv")
METRICS_PATH = Path("data/evaluation_metrics.json")

INPUT_COLUMN = "INPUT:input"
GOLDEN_COLUMN = "GOLDEN:agent"

LOG_EVERY = 10


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


def get_target_message_id(data: dict[str, Any]) -> str | None:
    """Извлекает ID целевого сообщения из исходного JSON."""

    for message in data.get("answer", []):
        if str(message.get("Ключ для разметки")) == "1":
            return message.get("ID запроса")

    return None


def process_dialog(
    raw_data: dict[str, Any],
    decision_tree: DecisionTree,
) -> dict[str, Any]:
    """Обрабатывает один диалог через DecisionTree."""

    input_data = InputData.model_validate(raw_data)

    context = build_context(
        input_data,
        messages_before=None, 
        messages_after=None
    )

    result = decision_tree.run(context)

    return result.model_dump(mode="json")


def evaluate_row(
    row: pd.Series,
    decision_tree: DecisionTree,
    row_index: int,
) -> dict[str, Any]:
    """Обрабатывает одну строку датасета и формирует результат."""

    golden = row.get(GOLDEN_COLUMN)
    filename = row.get("FILENAME")

    result = {
        "row_index": row_index,
        "message_id": None,
        "filename": filename,
        "golden": golden,
        "prediction": None,
        "is_correct": None,
        "spam": None,
        "spam_label": None,
        "spam_reason": None,
        "correct_label": None,
        "correct_reason": None,
        "category_label": None,
        "category_reason": None,
        "workflow_status": "failed",
        "error": None,
    }

    try:
        raw_data = json.loads(row[INPUT_COLUMN])

        result["message_id"] = get_target_message_id(raw_data)

        workflow_result = process_dialog(
            raw_data=raw_data,
            decision_tree=decision_tree,
        )

        result.update({
            "prediction": workflow_result["label"],
            "spam": workflow_result["spam"],
            "spam_label": workflow_result["decisions"]["spam"]["label"],
            "spam_reason": workflow_result["decisions"]["spam"]["reason"],
            "correct_label": workflow_result["decisions"]["correct"]["label"],
            "correct_reason": workflow_result["decisions"]["correct"]["reason"],
            "category_label": workflow_result["decisions"]["category"]["label"],
            "category_reason": workflow_result["decisions"]["category"]["reason"],
            "workflow_status": workflow_result["workflow"]["status"],
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


def calculate_metrics(results_df: pd.DataFrame) -> dict[str, Any]:
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

    correct_count = int(comparable["is_correct"].sum())

    accuracy = (
        correct_count / len(comparable)
        if len(comparable) > 0
        else None
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
    }


def main() -> None:
    OUTPUT_PATH.parent.mkdir(
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

    logger.info("Загружено строк: %s", len(df))

    # 2. Создаём LLM-клиент один раз
    llm_client = GigaChatClient()

    decision_model = GigaChatDecisionModel(
        client=llm_client,
    )

    decision_tree = DecisionTree(
        model=decision_model,
    )

    # 3. Последовательно обрабатываем датасет
    results = []

    for index, row in df.iterrows():

        result = evaluate_row(
            row=row,
            decision_tree=decision_tree,
            row_index=index,
        )

        results.append(result)

        processed = len(results)

        if processed % LOG_EVERY == 0 or processed == len(df):
            logger.info(
                "Обработано %s/%s",
                processed,
                len(df),
            )

            # Промежуточное сохранение
            pd.DataFrame(results).to_csv(
                OUTPUT_PATH,
                index=False,
                encoding="utf-8-sig",
            )

    # 4. Сохраняем итоговую таблицу
    results_df = pd.DataFrame(results)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # 5. Рассчитываем метрики
    metrics = calculate_metrics(results_df)

    METRICS_PATH.write_text(
        json.dumps(
            metrics,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    logger.info("Обработка завершена.")
    logger.info(
        "Accuracy: %s",
        metrics["accuracy"],
    )
    logger.info(
        "Результаты: %s",
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()
