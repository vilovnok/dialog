
from pathlib import Path


# GigaChat connection
GIGACHAT_BASE_URL = "https://gigachat-ift.sberdevices.delta.sbrf.ru/v1"
GIGACHAT_MODEL = "Qwen3.5-397b"

CA_BUNDLE_FILE = Path("/path/to/serv.pem")
CERT_FILE = Path("/path/to/client.pem")
KEY_FILE = Path("/path/to/client.key")

# Generation parameters
TEMPERATURE = 0.0
TIMEOUT = 600
REPETITION_PENALTY = 1.05
MAX_RETRIES = 0
