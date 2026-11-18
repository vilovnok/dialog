
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


# Допустимые статусы выполнения этапов
StepStatus = Literal[
    "pending",
    "completed",
    "skipped",
    "failed",
]


# --------------------------------------------------
# RAW INPUT DATA
# --------------------------------------------------

class RawMessage(BaseModel):
    """Сообщение в исходном формате датасета."""

    model_config = ConfigDict(populate_by_name=True)

    session_id: str = Field(alias="ID чат-сессии")
    channel_type: str | None = Field(
        default=None,
        alias="Тип канала",
    )
    split_name: str | None = Field(
        default=None,
        alias="Название сплита",
    )
    text: str = Field(alias="Текст сообщения")
    timestamp: str | None = Field(
        default=None,
        alias="Дата сообщения",
    )
    source: str = Field(alias="Источник сообщения")
    scenario: str | None = Field(
        default=None,
        alias="Сценарий",
    )
    agent: str | None = Field(
        default=None,
        alias="Агент",
    )
    annotation: str | None = Field(
        default=None,
        alias="Разметка",
    )
    target_key: str = Field(
        alias="Ключ для разметки",
    )
    request_id: str | None = Field(
        default=None,
        alias="ID запроса",
    )


class InputData(BaseModel):
    """Входная запись датасета."""

    answer: list[RawMessage]
    spam: bool | None = None


# --------------------------------------------------
# NORMALIZED DIALOG CONTEXT
# --------------------------------------------------

class Message(BaseModel):
    """Нормализованное сообщение для классификации."""

    message_id: str
    text: str
    source: str

    timestamp: str | None = None
    scenario: str | None = None
    agent: str | None = None


class DialogContext(BaseModel):
    session_id: str

    history_before_target: list[Message] = Field(
        default_factory=list
    )

    target_message: Message

    history_after_target: list[Message] = Field(
        default_factory=list
    )


# --------------------------------------------------
# CLASSIFICATION RESULTS
# --------------------------------------------------

class QuestionsConsecutiveResult(BaseModel):
    decision: Literal[
        "questions_consecutive",
        "not_questions_consecutive",
    ]
    reason: str

class DecisionResult(BaseModel):
    decision: Literal[
        "spam",
        "not_spam",
        "correct",
        "no_correct",
    ]
    reason: str


class CategoryResult(BaseModel):
    category_type: Literal[
        "agent",
        "special",
        "other",
    ]
    label: str
    reason: str


# --------------------------------------------------
# WORKFLOW STATE
# --------------------------------------------------

class StepResult(BaseModel):
    status: StepStatus = "pending"
    label: str | None = None
    reason: str | None = None


class WorkflowDecisions(BaseModel):
    spam: StepResult = Field(default_factory=StepResult)
    file: StepResult = Field(default_factory=StepResult)
    questions_consecutive: StepResult = Field(
        default_factory=StepResult
    )
    correct: StepResult = Field(default_factory=StepResult)
    category: StepResult = Field(default_factory=StepResult)


class WorkflowMetadata(BaseModel):
    current_step: str = "spam_check"

    status: Literal[
        "pending",
        "running",
        "completed",
        "failed",
    ] = "pending"

    final_label: str | None = None
    spam: bool = False


class WorkflowResult(BaseModel):
    session_id: str
    target_message_id: str

    spam: bool = False
    label: str | None = None

    decisions: WorkflowDecisions = Field(
        default_factory=WorkflowDecisions
    )

    workflow: WorkflowMetadata = Field(
        default_factory=WorkflowMetadata
    )
