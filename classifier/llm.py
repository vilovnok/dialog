
import os
import requests

from crowd_sdk.tagme import TagmeClientAdvanced


ENDPOINTS = {
    "direct": "https://tagme.sberdevices.ru/ai/llm/v1/chat/completions",
    "aihf": "https://tagme.sberdevices.ru/x/aihf/llm/v1/chat/completions",
}

CFG = {
    "direct": "~/configs/.ml_config.cfg",
    "aihf": "~/configs/.ml_config.cfg",
}

DEFAULT_MODEL = "DeepSeek-V4.1-Flash"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TOP_P = 1.0
MAX_TOKENS = 8000
TIMEOUT = (60, 180)


class GigaChatClient:

    def __init__(
        self,
        provider: str = "aihf",
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
    ):
        if provider not in ENDPOINTS:
            raise ValueError(
                f"Неизвестный provider: {provider}. "
                f"Допустимые значения: {list(ENDPOINTS)}"
            )

        self.provider = provider
        self.url = ENDPOINTS[provider]
        self.model = model
        self.temperature = temperature
        self.top_p = top_p

        self.token = self._get_token()

        self.session = requests.Session()

    def _get_token(self) -> str:
        cfg_path = os.path.expanduser(
            CFG[self.provider]
        )

        client = TagmeClientAdvanced(cfg_path)

        return client.client.token

    def invoke(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": MAX_TOKENS,
            "profanity_check": False,
        }

        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

        response = self.session.post(
            self.url,
            headers=headers,
            json=payload,
            verify=False,
            timeout=TIMEOUT,
        )

        # Обновляем токен при истечении срока действия.
        if response.status_code == 401:
            self.token = self._get_token()
            headers["Authorization"] = (
                f"Bearer {self.token}"
            )

            response = self.session.post(
                self.url,
                headers=headers,
                json=payload,
                verify=False,
                timeout=TIMEOUT,
            )

        response.raise_for_status()

        data = response.json()
        choices = data.get("choices") or []

        if not choices:
            raise ValueError(
                f"Модель вернула пустой choices: {data}"
            )

        content = (
            choices[0].get("message") or {}
        ).get("content", "")

        if isinstance(content, list):
            content = " ".join(
                part.get("text", "")
                for part in content
                if isinstance(part, dict)
            )

        if not isinstance(content, str) or not content.strip():
            raise ValueError(
                f"Модель вернула пустой ответ: {data}"
            )

        return content
