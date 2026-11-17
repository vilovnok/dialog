# Хорошо! Теперь мы обновили context_builder.py и теперь можем сами регулировать какое количество сообщений до и после целевого сообщения будет восприниматься моделью. 
import json
import re
from typing import Protocol

from classifier.llm import GigaChatClient
from classifier.schemas import (
    DialogContext,
    DecisionResult,
    CategoryResult,
)
from classifier.categories import (
    AGENT_CATEGORIES,
    SPECIAL_CATEGORIES,
    OTHER_CATEGORY,
)


class DecisionModel(Protocol):

    def check_spam(
        self,
        context: DialogContext,
    ) -> DecisionResult:
        ...

    def check_correct(
        self,
        context: DialogContext,
    ) -> DecisionResult:
        ...

    def classify_category(
        self,
        context: DialogContext,
    ) -> CategoryResult:
        ...


class GigaChatDecisionModel:

    def __init__(self, client: GigaChatClient):
        self.client = client

    # --------------------------------------------------
    # SPAM CHECK
    # --------------------------------------------------

    def check_spam(
        self,
        context: DialogContext,
    ) -> DecisionResult:

        system_prompt = """
# РОЛЬ

Ты — классификатор сообщений клиентов банковской организации.

# ЦЕЛЬ

Определить, является ли целевое сообщение клиента спамом.

# СТРУКТУРА ВХОДНЫХ ДАННЫХ

Входные данные представлены в формате JSON и содержат три основных блока:

- `history_before_target` — сообщения диалога, отправленные до целевого сообщения.
- `target_message` — целевое сообщение клиента, подлежащее классификации.
- `history_after_target` — сообщения диалога, отправленные после целевого сообщения.

Внутри объекта `target_message` присутствует поле `is_target: true`, явно обозначающее сообщение, которое необходимо классифицировать.

# ПРАВИЛА КЛАССИФИКАЦИИ

## 1. Спам (`spam`)

Относи сообщение к спаму, если оно содержит:

- бессмысленный набор символов;
- бессвязный текст, не содержащий осмысленного обращения;
- только эмодзи, не несущие осмысленной коммуникационной нагрузки;
- троллинг и провокационные сообщения, не содержащие реального обращения по существу;
- ненормативную лексику, не имеющую отношения к банковскому вопросу и не содержащую осмысленного обращения.

## 2. Не спам (`not_spam`)

Не считай сообщение спамом, если оно содержит:

- эмоциональное или негативное обращение;
- жалобу на банковские продукты и услуги;
- ошибки, опечатки или нарушения грамматики;
- короткий, но осмысленный вопрос;
- ненормативную лексику, используемую в контексте реального банковского обращения.

# ОБЪЕКТ АНАЛИЗА И ИСПОЛЬЗОВАНИЕ КОНТЕКСТА

1. Классифицируй исключительно сообщение из `target_message`.

2. Используй `history_before_target` и `history_after_target` только для понимания контекста целевого сообщения, если это необходимо.

3. Не классифицируй сообщения из истории отдельно.

4. Не переноси признаки спама из предыдущих или последующих сообщений на целевое сообщение.

5. Если целевое сообщение само по себе содержит осмысленное обращение, не признавай его спамом только из-за негативного тона, эмоциональности или особенностей формулировки.

6. Если контекста недостаточно, оценивай содержание целевого сообщения самостоятельно. Не додумывай отсутствующий смысл.

# ФОРМАТ ОТВЕТА

Верни результат строго в формате JSON:

{
    "decision": "spam" или "not_spam",
    "reason": "Краткое объяснение решения"
}

Поле `decision` должно содержать только одно из двух значений: `spam` или `not_spam`.
В поле `reason` укажи краткое объяснение, основанное на содержании целевого сообщения.
Не добавляй Markdown, комментарии или текст за пределами JSON.
""".strip()

        user_prompt = self._serialize_spam_context(context)
        print(f"User prompt для spam check: \n\n{user_prompt}\n***\n")

        result = self._invoke_json(
            system_prompt,
            user_prompt,
        )

        return DecisionResult.model_validate(result)

    # --------------------------------------------------
    # CORRECT CHECK
    # --------------------------------------------------

    def check_correct(
        self,
        context: DialogContext,
    ) -> DecisionResult:

        system_prompt = """
Ты — эксперт по оценке обоснованности переспросов AI-агентов
в банковских диалогах.

Твоя задача — определить, был ли переспрос агента действительно
необходим для понимания исходного обращения клиента.

Обязательно учитывай:
1. Историю диалога до целевого сообщения.
2. Целевое сообщение клиента.
3. Сообщение агента, содержащее переспрос.

Переспрос считается обоснованным (correct), если исходного
контекста недостаточно для однозначного понимания запроса
и агенту действительно необходимо получить дополнительную
информацию от клиента.

Переспрос считается необоснованным (no_correct), если:
- смысл обращения уже понятен из контекста;
- необходимая информация уже содержится в предыдущих сообщениях;
- агент переспрашивает о том, что клиент уже сообщил;
- смысл обращения можно определить без дополнительных уточнений;
- вопрос агента не помогает устранить реальную неоднозначность.

Не оценивай вежливость, стиль и формулировку переспроса.
Оценивай исключительно необходимость получения уточнения.

Важно:
Не используй информацию из ответов клиента, полученных
после переспроса. Оценивай ситуацию так, как если бы ты
находился на месте агента в момент формирования вопроса.

Верни результат строго в формате JSON:

{
    "decision": "correct" или "no_correct",
    "reason": "Краткое объяснение решения"
}

Не добавляй Markdown и текст вне JSON.
"""

        user_prompt = self._serialize_context(context)

        result = self._invoke_json(
            system_prompt,
            user_prompt,
        )

        return DecisionResult.model_validate(result)

    # --------------------------------------------------
    # CATEGORY CLASSIFICATION
    # --------------------------------------------------

    def classify_category(
        self,
        context: DialogContext,
    ) -> CategoryResult:

        agent_categories = sorted(AGENT_CATEGORIES)
        special_categories = sorted(SPECIAL_CATEGORIES)

        system_prompt = f"""
Ты — классификатор обращений клиентов банковской организации.

Твоя задача — определить категорию исходного обращения клиента.

Выбери ровно одну категорию из представленного списка.

ПРОФИЛЬНЫЕ АГЕНТЫ:
{json.dumps(agent_categories, ensure_ascii=False, indent=2)}

СПЕЦИАЛЬНЫЕ КЛАССЫ:
{json.dumps(special_categories, ensure_ascii=False, indent=2)}

ДРУГОЕ:
"{OTHER_CATEGORY}"

Правила классификации:

1. category_type = "agent", если обращение относится
   к одному из профильных агентов.

2. category_type = "special", если обращение относится
   к одному из специальных классов.

3. category_type = "other", если обращение не подходит
   ни под одну из перечисленных категорий.

4. label должен в точности совпадать с названием категории
   из соответствующего списка.

5. Не придумывай новые категории и не изменяй названия.

6. Анализируй смысл обращения, а не только отдельные слова.

7. Учитывай историю диалога и дополнительные сообщения клиента,
   если они помогают понять смысл исходного обращения.

Верни результат строго в формате JSON:

{{
    "category_type": "agent", "special" или "other",
    "label": "Название выбранной категории",
    "reason": "Краткое объяснение решения"
}}

Не добавляй Markdown и текст вне JSON.
"""

        user_prompt = self._serialize_context(context)

        result = self._invoke_json(
            system_prompt,
            user_prompt,
        )

        return CategoryResult.model_validate(result)

    # --------------------------------------------------
    # COMMON METHODS
    # --------------------------------------------------
    @staticmethod
    def _serialize_context(
        context: DialogContext,
    ) -> str:
        return context.model_dump_json(
            indent=2,
            exclude_none=True,
        )

    def _invoke_json(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> dict:

        response = self.client.invoke(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        return self._parse_json(response)


    @staticmethod
    def _serialize_spam_context(
        context: DialogContext,
    ) -> str:

        def serialize_message(
            message,
            is_target: bool = False,
        ) -> dict:

            result = {
                "message_id": message.message_id,
                "text": message.text,
                "source": message.source,
            }

            if is_target:
                result["is_target"] = True

            return result

        payload = {
            "history_before_target": [
                serialize_message(message)
                for message in context.history_before_target
            ],
            "target_message": serialize_message(
                context.target_message,
                is_target=True,
            ),
            "history_after_target": [
                serialize_message(message)
                for message in context.history_after_target
            ],
        }

        return json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )


    @staticmethod
    def _parse_json(response: str) -> dict:

        text = response.strip()

        # Удаляем Markdown-обёртку, если модель её добавила.
        text = re.sub(
            r"^```(?:json)?\s*",
            "",
            text,
        )
        text = re.sub(
            r"\s*```$",
            "",
            text,
        )

        try:
            result = json.loads(text)

        except json.JSONDecodeError:

            # Пытаемся извлечь JSON-объект из дополнительного текста.
            match = re.search(
                r"\{.*\}",
                text,
                re.DOTALL,
            )

            if match is None:
                raise ValueError(
                    f"Модель вернула некорректный JSON: {response}"
                )

            result = json.loads(match.group())

        if not isinstance(result, dict):
            raise ValueError(
                "Ожидался JSON-объект от модели."
            )

        return result
