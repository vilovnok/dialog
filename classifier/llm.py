from langchain_gigachat.chat_models import GigaChat

from config import (
    CA_BUNDLE_FILE,
    CERT_FILE,
    KEY_FILE,
    GIGACHAT_BASE_URL,
    GIGACHAT_MODEL,
    TEMPERATURE,
    TIMEOUT,
    REPETITION_PENALTY,
    MAX_RETRIES,
)


class GigaChatClient:

    def __init__(self):
        self.client = GigaChat(
            ca_bundle_file=str(CA_BUNDLE_FILE),
            cert_file=str(CERT_FILE),
            key_file=str(KEY_FILE),
            base_url=GIGACHAT_BASE_URL,
            model=GIGACHAT_MODEL,
            verify_ssl_certs=True,
            temperature=TEMPERATURE,
            timeout=TIMEOUT,
            repetition_penalty=REPETITION_PENALTY,
            max_retries=MAX_RETRIES,
        )

    def invoke(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:

        response = self.client.invoke(
            [
                ("system", system_prompt),
                ("human", user_prompt),
            ]
        )

        return response.content
