import json
import pandas as pd

from config import INPUT_PATH, INPUT_COLUMN

from classifier.schemas import InputData
from classifier.context_builder import build_context
from classifier.llm import GigaChatClient
from classifier.decision_model import GigaChatDecisionModel
from classifier.decision_tree import DecisionTree


def process_dialog(data: dict) -> dict:
    input_data = InputData.model_validate(data)

    context = build_context(
        input_data,
        messages_before=None,
        messages_after=None,
    )

    llm_client = GigaChatClient()

    decision_model = GigaChatDecisionModel(
        client=llm_client,
    )

    decision_tree = DecisionTree(
        model=decision_model,
    )

    result = decision_tree.run(context)

    return result.model_dump(mode="json")


def main():
    df = pd.read_csv(
        INPUT_PATH,
        encoding="utf-8-sig",
        dtype=str,
    )

    if INPUT_COLUMN not in df.columns:
        raise ValueError(
            f"Не найден столбец {INPUT_COLUMN}"
        )

    if df.empty:
        raise ValueError("Датасет пуст.")

    row = df.iloc[0]

    input_data = json.loads(
        row[INPUT_COLUMN]
    )

    result = process_dialog(input_data)

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()