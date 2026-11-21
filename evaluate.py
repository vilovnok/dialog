
import argparse
import json
import logging
import threading
import time

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from classifier.llm import GigaChatClient
from classifier.decision_model import GigaChatDecisionModel
from classifier.decision_tree import DecisionTree
from classifier.schemas import InputData
from classifier.context_builder import build_context

from classifier.prompts import get_prompt_metadata

from config import (
    INPUT_PATH,
    OUTPUT_DIR,
    INPUT_COLUMN,
    GOLDEN_COLUMN,
    FILENAME_COLUMN,
    PROVIDER,
    MODEL,
    LOG_EVERY,
    SAVE_EVERY,
)


# --------------------------------------------------
# GLOBAL SETTINGS
# --------------------------------------------------

worker_local = threading.local()

logger = logging.getLogger(__name__)


# --------------------------------------------------
# LOGGING
# --------------------------------------------------

def setup_logging(log_path: Path) -> None:
    """Настраивает вывод логов в консоль и файл."""

    log_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(threadName)s | %(message)s",
        handlers=[
            logging.FileHandler(
                log_path,
                encoding="utf-8",
            ),
            logging.StreamHandler(),
        ],
        force=True,
    )


# --------------------------------------------------
# WORKER INITIALIZATION
# --------------------------------------------------

def get_worker_runtime() -> tuple[DecisionTree, GigaChatClient]:
    """
    Создаёт отдельные объекты для каждого потока.

    Один поток использует собственные:
    - GigaChatClient;
    - GigaChatDecisionModel;
    - DecisionTree.

    При повторных вызовах объекты переиспользуются.
    """

    if not hasattr(worker_local, "decision_tree"):

        llm_client = GigaChatClient()

        decision_model = GigaChatDecisionModel(
            client=llm_client,
        )

        decision_tree = DecisionTree(
            model=decision_model,
        )

        worker_local.llm_client = llm_client
        worker_local.decision_tree = decision_tree

        logger.info(
            "Создан DecisionTree для потока %s",
            threading.current_thread().name,
        )

    return (
        worker_local.decision_tree,
        worker_local.llm_client,
    )


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
    """Обрабатывает один диалог через DecisionTree."""

    input_data = InputData.model_validate(raw_data)

    context = build_context(
        input_data,
        messages_before=None,
        messages_after=None,
    )

    result = decision_tree.run(context)

    return result.model_dump(mode="json")


# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def get_decision_value(
    decisions: dict[str, Any],
    decision_name: str,
    field: str,
) -> Any:
    """Безопасно извлекает поле из отдельного решения."""

    decision = decisions.get(decision_name) or {}

    return decision.get(field)


def get_stats_delta(
    before: dict[str, Any],
    after: dict[str, Any],
) -> dict[str, Any]:
    """Вычисляет разницу между двумя снимками статистики."""

    return {
        key: after.get(key, 0) - before.get(key, 0)
        for key in after
    }


# --------------------------------------------------
# ROW EVALUATION
# --------------------------------------------------

def evaluate_row(
    row: pd.Series,
    row_index: int,
    decision_tree: DecisionTree,
    llm_client: GigaChatClient,
) -> dict[str, Any]:
    """Обрабатывает одну строку датасета."""

    golden = row.get(GOLDEN_COLUMN)
    filename = row.get(FILENAME_COLUMN)

    result = {
        "row_index": row_index,
        "message_id": None,
        "filename": filename,

        "golden": golden,
        "prediction": None,
        "is_correct": None,

        "session_id": None,
        "spam": None,

        "spam_label": None,
        "spam_reason": None,

        "file_label": None,
        "file_reason": None,

        "questions_consecutive_label": None,
        "questions_consecutive_reason": None,

        "correct_label": None,
        "correct_reason": None,

        "category_label": None,
        "category_reason": None,

        "workflow_status": "failed",
        "workflow_current_step": None,

        "processing_time_sec": None,

        "llm_requests": 0,
        "llm_requests_successful": 0,
        "llm_requests_failed": 0,
        "llm_latency_sec": 0.0,

        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,

        "error": None,
    }

    started_at = time.perf_counter()
    stats_before = llm_client.get_stats()

    try:
        raw_data = json.loads(row[INPUT_COLUMN])

        result["message_id"] = get_target_message_id(
            raw_data
        )

        workflow_result = process_dialog(
            raw_data=raw_data,
            decision_tree=decision_tree,
        )

        decisions = workflow_result.get("decisions") or {}
        workflow = workflow_result.get("workflow") or {}

        result.update({
            "prediction": workflow_result.get("label"),
            "session_id": workflow_result.get("session_id"),
            "spam": workflow_result.get("spam"),

            "spam_label": get_decision_value(
                decisions, "spam", "label"
            ),
            "spam_reason": get_decision_value(
                decisions, "spam", "reason"
            ),

            "file_label": get_decision_value(
                decisions, "file", "label"
            ),
            "file_reason": get_decision_value(
                decisions, "file", "reason"
            ),

            "questions_consecutive_label": get_decision_value(
                decisions, "questions_consecutive", "label"
            ),
            "questions_consecutive_reason": get_decision_value(
                decisions, "questions_consecutive", "reason"
            ),

            "correct_label": get_decision_value(
                decisions, "correct", "label"
            ),
            "correct_reason": get_decision_value(
                decisions, "correct", "reason"
            ),

            "category_label": get_decision_value(
                decisions, "category", "label"
            ),
            "category_reason": get_decision_value(
                decisions, "category", "reason"
            ),

            "workflow_status": workflow.get(
                "status",
                "unknown",
            ),
            "workflow_current_step": workflow.get(
                "current_step"
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

    finally:
        result["processing_time_sec"] = (
            time.perf_counter() - started_at
        )

        stats_after = llm_client.get_stats()
        stats_delta = get_stats_delta(
            stats_before,
            stats_after,
        )

        result.update({
            "llm_requests": stats_delta["requests_total"],
            "llm_requests_successful": (
                stats_delta["requests_successful"]
            ),
            "llm_requests_failed": (
                stats_delta["requests_failed"]
            ),
            "llm_latency_sec": stats_delta["latency_sec"],

            "prompt_tokens": stats_delta["prompt_tokens"],
            "completion_tokens": stats_delta["completion_tokens"],
            "total_tokens": stats_delta["total_tokens"],
        })

    return result


def evaluate_row_parallel(
    row: pd.Series,
    row_index: int,
) -> dict[str, Any]:
    """Обрабатывает строку в текущем потоке."""

    decision_tree, llm_client = get_worker_runtime()

    return evaluate_row(
        row=row,
        row_index=row_index,
        decision_tree=decision_tree,
        llm_client=llm_client,
    )


# --------------------------------------------------
# METRICS
# --------------------------------------------------

def calculate_metrics(
    results_df: pd.DataFrame,
    workers: int,
    total_time_sec: float,
    prompt_metadata: dict,
) -> dict[str, Any]:
    """Рассчитывает метрики инференса."""

    total = len(results_df)

    successful = results_df[
        results_df["workflow_status"] == "completed"
    ]

    failed = results_df[
        results_df["workflow_status"] == "failed"
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

    def sum_column(column: str) -> int:
        return int(
            pd.to_numeric(
                results_df[column],
                errors="coerce",
            ).fillna(0).sum()
        )

    processing_times = pd.to_numeric(
        results_df["processing_time_sec"],
        errors="coerce",
    ).dropna()

    llm_latency = pd.to_numeric(
        results_df["llm_latency_sec"],
        errors="coerce",
    ).dropna()

    return {
        "run": {
            "timestamp": datetime.now().isoformat(),
            "provider": PROVIDER,
            "model": MODEL,
            "workers": workers,
            "total_time_sec": round(total_time_sec, 3),
        },

        "dataset": {
            "total_rows": total,
            "completed_rows": len(successful),
            "failed_rows": len(failed),
        },

        "classification": {
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
            "spam_rows": int(
                results_df["spam"].fillna(False).astype(bool).sum()
            ),
        },

        "performance": {
            "mean_processing_time_sec": (
                round(float(processing_times.mean()), 3)
                if len(processing_times) > 0
                else None
            ),
            "mean_llm_latency_sec": (
                round(float(llm_latency.mean()), 3)
                if len(llm_latency) > 0
                else None
            ),
            "rows_per_second": (
                round(total / total_time_sec, 3)
                if total_time_sec > 0
                else None
            ),
        },

        "llm": {
            "requests_total": sum_column("llm_requests"),
            "requests_successful": sum_column(
                "llm_requests_successful"
            ),
            "requests_failed": sum_column(
                "llm_requests_failed"
            ),
            "prompt_tokens": sum_column("prompt_tokens"),
            "completion_tokens": sum_column(
                "completion_tokens"
            ),
            "total_tokens": sum_column("total_tokens"),
        },
        "prompts": prompt_metadata,
    }


# --------------------------------------------------
# ARGUMENTS
# --------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Параллельный инференс классификатора."
    )

    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=1,
        help="Количество параллельных воркеров.",
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=INPUT_PATH,
        help="Путь к входному CSV.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Директория для результатов.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Ограничение количества строк для тестового запуска.",
    )

    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers должен быть не меньше 1.")

    if args.limit is not None and args.limit < 1:
        parser.error("--limit должен быть не меньше 1.")

    return args


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main() -> None:

    args = parse_args()

    run_id = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    output_dir = args.output_dir

    results_dir = output_dir / "results"
    metrics_dir = output_dir / "metrics"
    logs_dir = output_dir / "logs"

    for directory in (
        results_dir,
        metrics_dir,
        logs_dir,
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    results_path = (
        results_dir / f"evaluation_{run_id}.csv"
    )

    metrics_path = (
        metrics_dir / f"metrics_{run_id}.json"
    )

    log_path = (
        logs_dir / f"evaluation_{run_id}.log"
    )

    setup_logging(log_path)

    workers = args.workers

    logger.info("Начало инференса.")
    logger.info("Provider: %s", PROVIDER)
    logger.info("Model: %s", MODEL)
    logger.info("Workers: %s", workers)
    logger.info("Input: %s", args.input)

    # 1. Загружаем датасет.

    df = pd.read_csv(
        args.input,
        encoding="utf-8-sig",
        dtype=str,
    )

    required_columns = [
        INPUT_COLUMN,
        GOLDEN_COLUMN,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Не найдены обязательные колонки: {missing_columns}"
        )

    if args.limit is not None:
        df = df.head(args.limit)

    total_rows = len(df)

    if total_rows == 0:
        raise ValueError("В датасете нет строк для обработки.")

    logger.info("Загружено строк: %s", total_rows)

    # 2. Подготавливаем результаты.

    results = [None] * total_rows

    started_at = time.perf_counter()

    prompt_metadata = get_prompt_metadata()

    logger.info(
        "Используемые промпты: %s",
        json.dumps(
            prompt_metadata,
            ensure_ascii=False,
        ),
    )

    # 3. Запускаем параллельную обработку.

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

            # 4. Периодически сохраняем результаты.

            if (
                completed % SAVE_EVERY == 0
                or completed == total_rows
            ):
                completed_results = [
                    item
                    for item in results
                    if item is not None
                ]

                pd.DataFrame(
                    completed_results
                ).to_csv(
                    results_path,
                    index=False,
                    encoding="utf-8-sig",
                )

                logger.info(
                    "Промежуточно сохранено строк: %s",
                    len(completed_results),
                )

    total_time_sec = (
        time.perf_counter() - started_at
    )

    # 5. Итоговое сохранение.

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        results_path,
        index=False,
        encoding="utf-8-sig",
    )

    # 6. Рассчитываем метрики.

    metrics = calculate_metrics(
        results_df=results_df,
        workers=workers,
        total_time_sec=total_time_sec,
        prompt_metadata=prompt_metadata,
    )

    metrics_path.write_text(
        json.dumps(
            metrics,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # 7. Итоговые логи.

    logger.info("Инференс завершён.")
    logger.info(
        "Accuracy: %s",
        metrics["classification"]["accuracy"],
    )
    logger.info(
        "Completed: %s",
        metrics["dataset"]["completed_rows"],
    )
    logger.info(
        "Failed: %s",
        metrics["dataset"]["failed_rows"],
    )
    logger.info(
        "LLM requests: %s",
        metrics["llm"]["requests_total"],
    )
    logger.info(
        "Total time: %.2f sec",
        total_time_sec,
    )
    logger.info("Results: %s", results_path)
    logger.info("Metrics: %s", metrics_path)
    logger.info("Log: %s", log_path)


if __name__ == "__main__":
    main()
