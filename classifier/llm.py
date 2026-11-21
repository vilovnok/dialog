import os
import time
import threading
import requests

from crowd_sdk.tagme import TagmeClientAdvanced

from config import (
    PROVIDER,
    ENDPOINTS,
    CFG_PATHS,
    MODEL,
    TEMPERATURE,
    MAX_TOKENS,
    TIMEOUT,
    VERIFY_SSL,
)


class GigaChatClient:

    def __init__(self):
        if PROVIDER not in ENDPOINTS:
            raise ValueError(
                f"Неизвестный PROVIDER: {PROVIDER}"
            )

        self.url = ENDPOINTS[PROVIDER]
        self.model = MODEL
        self.token = self._get_token()

        self.session = requests.Session()

        self._stats_lock = threading.Lock()

        self.stats = {
            "requests_total": 0,
            "requests_successful": 0,
            "requests_failed": 0,
            "latency_sec": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    @staticmethod
    def _get_token() -> str:
        cfg_path = os.path.expanduser(
            CFG_PATHS[PROVIDER]
        )

        client = TagmeClientAdvanced(cfg_path)

        return client.client.token

    def get_stats(self) -> dict:
        """Возвращает снимок статистики клиента."""

        with self._stats_lock:
            return self.stats.copy()

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
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "profanity_check": False,
        }

        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

        started_at = time.perf_counter()

        with self._stats_lock:
            self.stats["requests_total"] += 1

        try:
            response = self.session.post(
                self.url,
                headers=headers,
                json=payload,
                verify=VERIFY_SSL,
                timeout=TIMEOUT,
            )

            # Обновляем токен и повторяем запрос один раз.
            if response.status_code == 401:
                self.token = self._get_token()

                headers["Authorization"] = (
                    f"Bearer {self.token}"
                )

                response = self.session.post(
                    self.url,
                    headers=headers,
                    json=payload,
                    verify=VERIFY_SSL,
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

            if (
                not isinstance(content, str)
                or not content.strip()
            ):
                raise ValueError(
                    f"Модель вернула пустой ответ: {data}"
                )

            usage = data.get("usage") or {}

            with self._stats_lock:
                self.stats["requests_successful"] += 1

                self.stats["prompt_tokens"] += (
                    usage.get("prompt_tokens") or 0
                )

                self.stats["completion_tokens"] += (
                    usage.get("completion_tokens") or 0
                )

                self.stats["total_tokens"] += (
                    usage.get("total_tokens") or 0
                )

            return content

        except Exception:
            with self._stats_lock:
                self.stats["requests_failed"] += 1

            raise

        finally:
            elapsed = time.perf_counter() - started_at

            with self._stats_lock:
                self.stats["latency_sec"] += elapsed