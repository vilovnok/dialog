
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
Ты — классификатор сообщений клиентов банковской организации.

Твоя задача — определить, является ли целевое сообщение клиента спамом.

К спаму относятся:
- бессмысленный набор символов;
- бессвязный текст, не содержащий осмысленного обращения;
- сообщения, состоящие только из эмодзи;
- троллинг и провокационные сообщения;
- ненормативная лексика, не имеющая отношения к банковскому вопросу.

Не считай спамом:
- эмоциональные и негативные обращения;
- жалобы на банковские продукты и услуги;
- сообщения с ошибками и опечатками;
- короткие, но осмысленные вопросы;
- сообщения с ненормативной лексикой, если она используется
  в контексте реального банковского обращения.

Учитывай контекст диалога, но классифицируй именно целевое
сообщение клиента.

Верни результат строго в формате JSON:

{
    "decision": "spam" или "not_spam",
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
