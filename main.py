
import json
import pandas as pd

from classifier.schemas import InputData
from classifier.context_builder import build_context
from classifier.llm import GigaChatClient
from classifier.decision_model import GigaChatDecisionModel
from classifier.decision_tree import DecisionTree


def process_dialog(data: dict) -> dict:
    # 1. Валидация исходных данных
    input_data = InputData.model_validate(data)

    # 2. Формирование контекста
    context = build_context(input_data)

    # 3. Инициализация модели
    llm_client = GigaChatClient()
    decision_model = GigaChatDecisionModel(client=llm_client)

    # 4. Запуск дерева решений
    decision_tree = DecisionTree(model=decision_model)

    result = decision_tree.run(context)

    return result.model_dump(mode="json")


def main():
    df = pd.read_csv(
        "/Users/richardgurtsiev/Desktop/projects/vibe_code/workspaces/data/tagme_balanced_small_dataset.csv",
        encoding="utf-8-sig",
        dtype=str,
    )

    # Берём первую запись
    row = df.iloc[0]

    # Извлекаем JSON из INPUT:input
    input_data = json.loads(row["INPUT:input"])

    # Запускаем классификацию
    result = process_dialog(input_data)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
