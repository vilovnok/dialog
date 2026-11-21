from pathlib import Path



ENDPOINTS = {
    "direct": "https://tagme.sberdevices.ru/ai/llm/v1/chat/completions",
    "aihf": "https://tagme.sberdevices.ru/x/aihf/llm/v1/chat/completions",
    "vsellm": "https://tagme.sberdevices.ru/x/ai/llm/v1/chat/completions",
}

CFG_PATHS = {
    "direct": "~/configs/.ml_config.cfg",
    "aihf": "~/configs/.ml_config.cfg",
    "vsellm": "~/configs/.ml_config.cfg",
}


MODEL = "DeepSeek-V4.1-Flash"
PROVIDER = "aihf" 
VERIFY_SSL = False

TEMPERATURE = 0.0
MAX_TOKENS = 8000
TIMEOUT = (60, 120)

INPUT_PATH = Path(
    "data/tagme_balanced_small_dataset.csv"
)

OUTPUT_DIR = Path("data/evaluation")

INPUT_COLUMN = "INPUT:input"
GOLDEN_COLUMN = "GOLDEN:agent"
FILENAME_COLUMN = "FILENAME"

LOG_EVERY = 10
SAVE_EVERY = 10