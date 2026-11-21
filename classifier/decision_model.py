
import json
import re

from typing import Protocol

from classifier.llm import GigaChatClient

from classifier.schemas import (
    DialogContext,
    DecisionResult,
    CategoryResult,
)

from classifier.prompts import (
    get_spam_prompt,
    get_correct_prompt,
    get_category_prompt,
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

        system_prompt = get_spam_prompt()

        user_prompt = self._serialize_spam_context(context)

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

        system_prompt = get_correct_prompt()

        user_prompt = self._serialize_correct_context(context)

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

        system_prompt = get_category_prompt()

        user_prompt = self._serialize_category_context(context)

        result = self._invoke_json(
            system_prompt,
            user_prompt,
        )

        return CategoryResult.model_validate(result)

    # --------------------------------------------------
    # COMMON METHODS
    # --------------------------------------------------

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

    # --------------------------------------------------
    # CONTEXT SERIALIZATION
    # --------------------------------------------------

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
    def _serialize_correct_context(
        context: DialogContext,
    ) -> str:

        return GigaChatDecisionModel._serialize_spam_context(
            context
        )

    @staticmethod
    def _serialize_category_context(
        context: DialogContext,
    ) -> str:

        return GigaChatDecisionModel._serialize_spam_context(
            context
        )

    # --------------------------------------------------
    # JSON PARSING
    # --------------------------------------------------

    @staticmethod
    def _parse_json(response: str) -> dict:

        text = response.strip()

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
