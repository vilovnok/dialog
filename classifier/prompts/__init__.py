
import hashlib
import json
import re

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from classifier.categories import (
    AGENT_CATEGORIES_WITH_DESCRIPTIONS,
    SPECIAL_CATEGORIES_WITH_DESCRIPTIONS,
)


PROMPTS_DIR = Path(__file__).parent

PROMPT_FILES = {
    "spam": "spam.md",
    "correct": "correct.md",
    "category": "category.md",
}

VERSION_PATTERN = re.compile(
    r"^\d+\.\d+\.\d+$"
)


@dataclass(frozen=True)
class PromptDocument:
    name: str
    version: str
    description: str
    content: str


def _parse_frontmatter(
    text: str,
) -> tuple[dict[str, str], str]:
    """Извлекает простые метаданные из Markdown."""

    match = re.match(
        r"\A---\s*\n(.*?)\n---\s*\n?(.*)\Z",
        text,
        re.DOTALL,
    )

    if match is None:
        raise ValueError(
            "Не найден YAML front matter в файле промпта."
        )

    metadata_text = match.group(1)
    content = match.group(2).strip()

    metadata = {}

    for line in metadata_text.splitlines():
        if not line.strip():
            continue

        if ":" not in line:
            raise ValueError(
                f"Некорректная строка метаданных: {line}"
            )

        key, value = line.split(":", 1)

        metadata[key.strip()] = value.strip()

    return metadata, content


@lru_cache(maxsize=None)
def load_prompt(
    name: str,
) -> PromptDocument:
    """Загружает и кэширует Markdown-промпт."""

    if name not in PROMPT_FILES:
        raise ValueError(
            f"Неизвестный промпт: {name}"
        )

    prompt_path = PROMPTS_DIR / PROMPT_FILES[name]

    if not prompt_path.exists():
        raise FileNotFoundError(
            f"Файл промпта не найден: {prompt_path}"
        )

    text = prompt_path.read_text(
        encoding="utf-8"
    )

    metadata, content = _parse_frontmatter(text)

    required_fields = {
        "name",
        "version",
        "description",
    }

    missing_fields = (
        required_fields - metadata.keys()
    )

    if missing_fields:
        raise ValueError(
            f"В {prompt_path.name} отсутствуют метаданные: "
            f"{sorted(missing_fields)}"
        )

    if metadata["name"] != name:
        raise ValueError(
            f"Имя промпта в {prompt_path.name} "
            f"не совпадает с ожидаемым: {name}"
        )

    if not VERSION_PATTERN.fullmatch(
        metadata["version"]
    ):
        raise ValueError(
            f"Некорректная версия промпта "
            f"{metadata['version']}"
        )

    if not content:
        raise ValueError(
            f"Промпт {prompt_path.name} пуст."
        )

    return PromptDocument(
        name=metadata["name"],
        version=metadata["version"],
        description=metadata["description"],
        content=content,
    )


def get_spam_prompt() -> str:
    return load_prompt("spam").content


def get_correct_prompt() -> str:
    return load_prompt("correct").content


def get_category_prompt() -> str:
    """Загружает промпт и подставляет категории."""

    prompt = load_prompt("category").content

    agent_categories = json.dumps(
        AGENT_CATEGORIES_WITH_DESCRIPTIONS,
        ensure_ascii=False,
        indent=2,
    )

    special_categories = json.dumps(
        SPECIAL_CATEGORIES_WITH_DESCRIPTIONS,
        ensure_ascii=False,
        indent=2,
    )

    prompt = prompt.replace(
        "{{AGENT_CATEGORIES_WITH_DESCRIPTIONS}}",
        agent_categories,
    )

    prompt = prompt.replace(
        "{{SPECIAL_CATEGORIES_WITH_DESCRIPTIONS}}",
        special_categories,
    )

    if (
        "{{AGENT_CATEGORIES_WITH_DESCRIPTIONS}}" in prompt
        or "{{SPECIAL_CATEGORIES_WITH_DESCRIPTIONS}}" in prompt
    ):
        raise ValueError(
            "Не все переменные промпта category были подставлены."
        )

    return prompt


def get_prompt_metadata() -> dict:
    """Возвращает версии и хеши фактически используемых промптов."""

    resolved_prompts = {
        "spam": get_spam_prompt(),
        "correct": get_correct_prompt(),
        "category": get_category_prompt(),
    }

    metadata = {}

    for name, content in resolved_prompts.items():
        document = load_prompt(name)

        metadata[name] = {
            "version": document.version,
            "description": document.description,
            "sha256": hashlib.sha256(
                content.encode("utf-8")
            ).hexdigest(),
        }

    return metadata
