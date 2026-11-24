import argparse
import hashlib
import json
import logging
import threading
import time
import uuid

from datetime import datetime

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


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

INPUT_PATH = Path(
    "/home/work/22850885@sigma.sbrf.ru/project/workplaces/data/tagme_balanced_small_dataset.csv"
)

INPUT_COLUMN = "INPUT:input"
GOLDEN_COLUMN = "GOLDEN:agent"

RUNS_DIR = Path("data/runs")
REGISTRY_PATH = RUNS_DIR / "runs_registry.csv"

LOG_EVERY = 10

MAX_RETRIES = 3
RETRY_DELAYS = (2, 4)

worker_local = threading.local()
registry_lock = threading.Lock()


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# --------------------------------------------------
# RUN METADATA
# --------------------------------------------------

def get_timestamp() -> str:
    """Возвращает локальное время с часовым поясом."""

    return datetime.now().astimezone().isoformat(
        timespec="seconds"
    )


def get_dataset_hash(path: Path) -> str:
    """Вычисляет SHA-256 исходного датасета."""

    sha256 = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            sha256.update(chunk)

    return sha256.hexdigest()


def save_registry_entry(
    entry: dict[str, Any],
) -> None:
    """Добавляет или обновляет запись в реестре запусков."""

    with registry_lock:

        REGISTRY_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if REGISTRY_PATH.exists():
            registry = pd.read_csv(
                REGISTRY_PATH,
                encoding="utf-8-sig",
            )
        else:
            registry = pd.DataFrame()

        if (
            not registry.empty
            and "run_id" in registry.columns
        ):
            registry = registry[
                registry["run_id"] != entry["run_id"]
            ]

        registry = pd.concat(
            [
                registry,
                pd.DataFrame([entry]),
            ],
            ignore_index=True,
        )

        registry.to_csv(
            REGISTRY_PATH,
            index=False,
            encoding="utf-8-sig",
        )


def save_run_metadata(
    metadata: dict[str, Any],
    metadata_path: Path,
) -> None:
    """Сохраняет метаданные конкретного запуска."""

    metadata_path.write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )


# --------------------------------------------------
# WORKER INITIALIZATION
# --------------------------------------------------

def get_worker_decision_tree(
    force_new: bool = False,
) -> DecisionTree:
    """Возвращает DecisionTree текущего потока.

    force_new=True создаёт новый клиент и дерево решений.
    Используется после ошибки предыдущей попытки.
    """

    if (
        force_new
        or not hasattr(worker_local, "decision_tree")
    ):
        llm_client = GigaChatClient()

        decision_model = GigaChatDecisionModel(
            client=llm_client,
        )

        worker_local.decision_tree = DecisionTree(
            model=decision_model,
        )

        logger.info(
            "Создан%s DecisionTree для потока %s",
            " заново" if force_new else "",
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
        if str(
            message.get("Ключ для разметки")
        ) == "1":
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
    """Определяет этап, сформировавший итоговую категорию."""

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
    run_id: str,
) -> dict[str, Any]:
    """Обрабатывает одну строку датасета."""

    golden = row.get(GOLDEN_COLUMN)
    filename = row.get("FILENAME")

    result = {
        "run_id": run_id,
        "record_id": f"{run_id}:{row_index}:no-message-id",
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
        "attempts": 1,
    }

    try:
        raw_data = json.loads(row[INPUT_COLUMN])

        result["message_id"] = get_target_message_id(
            raw_data
        )

        result["record_id"] = (
            f"{run_id}:{row_index}:"
            f"{result['message_id'] or 'no-message-id'}"
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

        final_source = get_final_source(workflow_result)

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

        if (
            pd.notna(golden)
            and result["prediction"] is not None
        ):
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
    run_id: str,
) -> dict[str, Any]:
    """Обрабатывает строку с повторными попытками."""

    last_result: dict[str, Any] | None = None

    for attempt in range(1, MAX_RETRIES + 1):

        try:
            decision_tree = get_worker_decision_tree(
                force_new=(attempt > 1),
            )

            row_result = evaluate_row(
                row=row,
                decision_tree=decision_tree,
                row_index=row_index,
                run_id=run_id,
            )

            row_result["attempts"] = attempt
            last_result = row_result

            if (
                row_result.get("error") is None
                and row_result.get("workflow_status") == "completed"
            ):
                if attempt > 1:
                    logger.info(
                        "Строка %s успешно обработана "
                        "с попытки %s/%s",
                        row_index,
                        attempt,
                        MAX_RETRIES,
                    )

                return row_result

            error_message = (
                row_result.get("error")
                or (
                    "workflow_status="
                    f"{row_result.get('workflow_status')}"
                )
            )

        except Exception as exc:
            error_message = str(exc)

            logger.exception(
                "Ошибка запуска обработки строки %s, "
                "попытка %s/%s",
                row_index,
                attempt,
                MAX_RETRIES,
            )

            last_result = {
                "run_id": run_id,
                "record_id": (
                    f"{run_id}:{row_index}:no-message-id"
                ),
                "row_index": row_index,
                "workflow_status": "failed",
                "error": error_message,
                "attempts": attempt,
            }

        if attempt < MAX_RETRIES:
            delay = RETRY_DELAYS[attempt - 1]

            logger.warning(
                "Строка %s завершилась ошибкой "
                "(попытка %s/%s): %s. Повтор через %s сек.",
                row_index,
                attempt,
                MAX_RETRIES,
                error_message,
                delay,
            )

            time.sleep(delay)

    logger.error(
        "Строка %s не обработана после %s попыток. "
        "Последняя ошибка: %s",
        row_index,
        MAX_RETRIES,
        (last_result or {}).get("error"),
    )

    return last_result or {
        "run_id": run_id,
        "record_id": f"{run_id}:{row_index}:no-message-id",
        "row_index": row_index,
        "workflow_status": "failed",
        "error": "Не удалось получить результат обработки",
        "attempts": MAX_RETRIES,
    }


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
        comparable["is_correct"].fillna(False).sum()
    )

    accuracy = (
        correct_count / len(comparable)
        if len(comparable) > 0
        else None
    )

    final_source_counts = (
        results_df["final_source"]
        .value_counts(dropna=False)
        .to_dict()
    )

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
        help="Количество параллельных воркеров.",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Название используемой модели.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Temperature для фиксации в метаданных.",
    )

    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Top-p для фиксации в метаданных.",
    )

    parser.add_argument(
        "--provider",
        type=str,
        choices=["auth", "direct"],
        required=True,
        help="Способ подключения к API: auth или direct.",
    )

    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers должен быть не меньше 1.")

    if (
        args.temperature is not None
        and not 0 <= args.temperature <= 2
    ):
        parser.error(
            "--temperature должен быть в диапазоне от 0 до 2."
        )

    if (
        args.top_p is not None
        and not 0 <= args.top_p <= 1
    ):
        parser.error(
            "--top-p должен быть в диапазоне от 0 до 1."
        )

    return args


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main() -> None:

    args = parse_args()

    # 1. Создаём идентификатор запуска.

    started_at = get_timestamp()
    start_time = time.monotonic()

    run_id = (
        datetime.now().strftime("%Y%m%d_%H%M%S")
        + "_"
        + uuid.uuid4().hex[:8]
    )

    run_dir = RUNS_DIR / run_id

    run_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    output_path = run_dir / "results.csv"
    metrics_path = run_dir / "metrics.json"
    metadata_path = run_dir / "run.json"

    # 2. Загружаем датасет.

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Датасет не найден: {INPUT_PATH}"
        )

    dataset_hash = get_dataset_hash(INPUT_PATH)

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
    workers = args.workers

    # 3. Формируем метаданные запуска.

    run_metadata = {
        "run_id": run_id,
        "status": "running",

        "started_at": started_at,
        "finished_at": None,
        "duration_seconds": None,

        "model_name": args.model_name,
        "provider": args.provider,
        "workers": workers,
        "temperature": args.temperature,
        "top_p": args.top_p,

        "dataset_path": str(INPUT_PATH.resolve()),
        "dataset_hash": dataset_hash,
        "total_rows": total_rows,

        "accuracy": None,
        "coverage": None,
        "completed_rows": 0,
        "failed_rows": 0,

        "results_path": str(output_path),
        "metrics_path": str(metrics_path),
    }

    save_run_metadata(
        run_metadata,
        metadata_path,
    )

    # Регистрируем запуск до начала обработки.

    registry_entry = run_metadata.copy()

    save_registry_entry(registry_entry)

    logger.info("=" * 60)
    logger.info("Запуск эксперимента: %s", run_id)
    logger.info("Модель: %s", args.model_name)
    logger.info("Workers: %s", workers)
    logger.info("Provider: %s", args.provider)
    logger.info("Temperature: %s", args.temperature)
    logger.info("Top-p: %s", args.top_p)
    logger.info("Датасет: %s", INPUT_PATH)
    logger.info("SHA-256: %s", dataset_hash)
    logger.info("Количество строк: %s", total_rows)
    logger.info("=" * 60)

    results = [None] * total_rows

    try:
        # 4. Параллельная обработка.

        with ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="llm-worker",
        ) as executor:

            futures = {
                executor.submit(
                    evaluate_row_parallel,
                    row,
                    position,
                    run_id,
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
                        "run_id": run_id,
                        "record_id": (
                            f"{run_id}:{position}:no-message-id"
                        ),
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
                        "Запуск %s: обработано %s/%s",
                        run_id,
                        completed,
                        total_rows,
                    )

                    completed_results = [
                        item
                        for item in results
                        if item is not None
                    ]

                    partial_df = pd.DataFrame(
                        completed_results
                    )

                    partial_df.to_csv(
                        output_path,
                        index=False,
                        encoding="utf-8-sig",
                    )

                    # Сохраняем промежуточные метрики
                    # и прогресс в реестр.

                    partial_metrics = calculate_metrics(
                        partial_df
                    )

                    registry_entry.update({
                        "completed_rows": (
                            partial_metrics["completed_rows"]
                        ),
                        "failed_rows": (
                            partial_metrics["failed_rows"]
                        ),
                        "accuracy": partial_metrics["accuracy"],
                        "coverage": partial_metrics["coverage"],
                    })

                    save_registry_entry(registry_entry)

        # 5. Сохраняем окончательные результаты.

        results_df = pd.DataFrame(results)

        results_df.to_csv(
            output_path,
            index=False,
            encoding="utf-8-sig",
        )

        # 6. Рассчитываем метрики.

        metrics = calculate_metrics(results_df)

        metrics.update({
            "run_id": run_id,
            "model_name": args.model_name,
            "provider": args.provider,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "workers": workers,
            "started_at": started_at,
            "dataset_hash": dataset_hash,
        })

        # 7. Сохраняем метрики.

        metrics_path.write_text(
            json.dumps(
                metrics,
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

        # 8. Обновляем статус запуска.

        finished_at = get_timestamp()
        duration_seconds = round(
            time.monotonic() - start_time,
            2,
        )

        run_status = (
            "completed"
            if metrics["failed_rows"] == 0
            else "completed_with_errors"
        )

        run_metadata.update({
            "status": run_status,
            "finished_at": finished_at,
            "duration_seconds": duration_seconds,

            "accuracy": metrics["accuracy"],
            "coverage": metrics["coverage"],

            "completed_rows": metrics["completed_rows"],
            "failed_rows": metrics["failed_rows"],
        })

        save_run_metadata(
            run_metadata,
            metadata_path,
        )

        registry_entry.update(run_metadata)

        save_registry_entry(registry_entry)

        logger.info("=" * 60)
        logger.info("Эксперимент завершён: %s", run_id)
        logger.info("Статус: %s", run_status)
        logger.info("Accuracy: %s", metrics["accuracy"])
        logger.info("Coverage: %s", metrics["coverage"])
        logger.info(
            "Успешно обработано: %s",
            metrics["completed_rows"],
        )
        logger.info(
            "Ошибок: %s",
            metrics["failed_rows"],
        )
        logger.info(
            "Время выполнения: %s сек.",
            duration_seconds,
        )
        logger.info("Результаты: %s", output_path)
        logger.info("Метрики: %s", metrics_path)
        logger.info("Метаданные: %s", metadata_path)
        logger.info("Реестр: %s", REGISTRY_PATH)
        logger.info("=" * 60)

    except Exception as exc:

        # 9. Фиксируем аварийное завершение запуска.

        finished_at = get_timestamp()

        run_metadata.update({
            "status": "failed",
            "finished_at": finished_at,
            "duration_seconds": round(
                time.monotonic() - start_time,
                2,
            ),
            "error": str(exc),
        })

        save_run_metadata(
            run_metadata,
            metadata_path,
        )

        registry_entry.update(run_metadata)

        save_registry_entry(registry_entry)

        logger.exception(
            "Эксперимент %s аварийно завершился.",
            run_id,
        )

        raise


if __name__ == "__main__":
    main()

# "GigaChat-3.5-432B-A28B"

# python evaluate.py \
#     --model-name "DeepSeek-V4.1-Flash" \
#     --provider auth \
#     --workers 20 \
#     --temperature 0.2 \
#     --top-p 0.9