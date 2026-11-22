
from classifier.categories import (
    CORRECT_CATEGORY,
    FILE_CATEGORY,
    QUESTIONS_CONSECUTIVE_CATEGORY,
    OTHER_CATEGORY,
    is_valid_category,
)

from classifier.schemas import (
    DialogContext,
    CategoryResult,
    StepResult,
    WorkflowResult,
)

from classifier.file_detector import check_file

from classifier.decision_model import (
    DecisionModel,
)


class DecisionTree:

    def __init__(self, model: DecisionModel):
        self.model = model

    def run(self, context: DialogContext) -> WorkflowResult:

        result = WorkflowResult(
            session_id=context.session_id,
            target_message_id=context.target_message.message_id,
        )

        result.workflow.status = "running"

        active_step = "spam_check"

        try:

            # --------------------------------------------------
            # 1. SPAM CHECK
            # --------------------------------------------------

            result.workflow.current_step = active_step

            spam_result = self.model.check_spam(context)

            result.decisions.spam = StepResult(
                status="completed",
                label=spam_result.decision,
                reason=spam_result.reason,
            )

            if spam_result.decision == "spam":

                result.spam = True
                result.workflow.spam = True

                result.decisions.file = StepResult(
                    status="skipped",
                    reason=(
                        "Проверка категории «Файл» не выполнялась: "
                        "сообщение определено как спам."
                    ),
                )

                result.decisions.questions_consecutive = StepResult(
                    status="skipped",
                    reason=(
                        "Проверка категории «Вопросы подряд» "
                        "не выполнялась: сообщение определено как спам."
                    ),
                )

                result.decisions.correct = StepResult(
                    status="skipped",
                    reason="Сообщение определено как спам.",
                )

                result.decisions.category = StepResult(
                    status="skipped",
                    reason="Классификация категории не выполнялась.",
                )

                return self._finish(result)

            # --------------------------------------------------
            # 2. FILE CHECK
            # --------------------------------------------------

            active_step = "file_check"
            result.workflow.current_step = active_step

            is_file = check_file(context)

            result.decisions.file = StepResult(
                status="completed",
                label=FILE_CATEGORY if is_file else "not_file",
                reason=(
                    "Обнаружен признак категории «Файл»."
                    if is_file
                    else (
                        "Условия определения категории «Файл» "
                        "не выполнены."
                    )
                ),
            )

            if is_file:

                result.label = FILE_CATEGORY

                result.decisions.questions_consecutive = StepResult(
                    status="skipped",
                    reason=(
                        "Проверка категории «Вопросы подряд» "
                        "не выполнялась: обращение отнесено "
                        "к категории «Файл»."
                    ),
                )

                result.decisions.correct = StepResult(
                    status="skipped",
                    reason="Обращение отнесено к категории «Файл».",
                )

                result.decisions.category = StepResult(
                    status="skipped",
                    reason="Классификация категории не выполнялась.",
                )

                return self._finish(result)

            # --------------------------------------------------
            # 3. QUESTIONS CONSECUTIVE CHECK
            # --------------------------------------------------

            active_step = "questions_consecutive_check"
            result.workflow.current_step = active_step

            has_consecutive = self._has_consecutive_client_messages(
                context
            )

            if has_consecutive:

                result.decisions.questions_consecutive = StepResult(
                    status="completed",
                    label="questions_consecutive",
                    reason=(
                        "Целевое сообщение клиента непосредственно "
                        "соседствует с другим сообщением клиента."
                    ),
                )

                result.label = QUESTIONS_CONSECUTIVE_CATEGORY

                result.decisions.correct = StepResult(
                    status="skipped",
                    reason=(
                        "Обращение отнесено к категории "
                        "«Вопросы подряд»."
                    ),
                )

                result.decisions.category = StepResult(
                    status="skipped",
                    reason="Классификация категории не выполнялась.",
                )

                return self._finish(result)

            result.decisions.questions_consecutive = StepResult(
                status="skipped",
                reason=(
                    "Структурное условие не выполнено: "
                    "целевое сообщение не принадлежит CLIENT "
                    "или непосредственно соседние сообщения "
                    "не принадлежат CLIENT."
                ),
            )

            # --------------------------------------------------
            # 4. CORRECT CHECK
            # --------------------------------------------------

            active_step = "correct_check"
            result.workflow.current_step = active_step

            correct_result = self.model.check_correct(context)

            result.decisions.correct = StepResult(
                status="completed",
                label=correct_result.decision,
                reason=correct_result.reason,
            )

            # Проверяем согласованность decision и category_group.

            if correct_result.decision == "correct":

                if correct_result.category_group != "no":
                    raise ValueError(
                        "Для decision='correct' "
                        "category_group должен быть 'no'. "
                        f"Получено: {correct_result.category_group}"
                    )

                result.label = CORRECT_CATEGORY

                result.decisions.category = StepResult(
                    status="skipped",
                    reason="Переспрос признан обоснованным.",
                )

                return self._finish(result)

            elif correct_result.decision == "no_correct":

                if correct_result.category_group not in (
                    "agent",
                    "special",
                    "no",
                ):
                    raise ValueError(
                        "Для decision='no_correct' необходимо указать "
                        "category_group: agent, special или no. "
                        f"Получено: {correct_result.category_group}"
                    )

            # --------------------------------------------------
            # 5. CATEGORY GROUP ROUTING
            # --------------------------------------------------

            if correct_result.category_group == "agent":

                active_step = "agent_classification"
                result.workflow.current_step = active_step

                category_result = self.model.classify_agent(context)

                self._validate_category(
                    category_result,
                    expected_type="agent",
                )

            elif correct_result.category_group == "special":

                active_step = "special_classification"
                result.workflow.current_step = active_step

                category_result = self.model.classify_special(context)

                self._validate_category(
                    category_result,
                    expected_type="special",
                )

            elif correct_result.category_group == "no":

                result.label = OTHER_CATEGORY

                result.decisions.category = StepResult(
                    status="completed",
                    label=OTHER_CATEGORY,
                    reason=(
                        "Предмет обращения не относится "
                        "ни к одной группе классификации."
                    ),
                )

                return self._finish(result)

            else:
                raise ValueError(
                    "Модель вернула неизвестную группу классификации: "
                    f"{correct_result.category_group}"
                )

            # --------------------------------------------------
            # 6. CATEGORY VALIDATION
            # --------------------------------------------------

            self._validate_category(category_result)

            result.decisions.category = StepResult(
                status="completed",
                label=category_result.label,
                reason=category_result.reason,
            )

            result.label = category_result.label

            return self._finish(result)

        except Exception as exc:

            self._mark_failed(
                result=result,
                step=active_step,
                error=exc,
            )

            raise

    # --------------------------------------------------
    # QUESTIONS CONSECUTIVE DETECTION
    # --------------------------------------------------

    @staticmethod
    def _has_consecutive_client_messages(
        context: DialogContext,
    ) -> bool:
        """
        Проверяет, соседствует ли целевое сообщение клиента
        с другим сообщением клиента в исходном диалоге.

        Информация о соседних сообщениях определяется
        в context_builder.py независимо от размера контекста.
        """

        if context.target_message.source != "CLIENT":
            return False

        return (
            context.has_previous_client_message
            or context.has_next_client_message
        )

    # --------------------------------------------------
    # VALIDATION
    # --------------------------------------------------

    @staticmethod
    def _validate_category(
        category: CategoryResult,
        expected_type: str,
    ) -> None:

        if category.category_type != expected_type:
            raise ValueError(
                "Классификатор вернул категорию неправильного типа: "
                f"expected={expected_type}, "
                f"actual={category.category_type}, "
                f"label={category.label}"
            )

        if not is_valid_category(
            category_type=category.category_type,
            label=category.label,
        ):
            raise ValueError(
                "Модель вернула недопустимую категорию: "
                f"type={category.category_type}, "
                f"label={category.label}"
            )

    # --------------------------------------------------
    # WORKFLOW STATUS
    # --------------------------------------------------

    @staticmethod
    def _finish(
        result: WorkflowResult,
    ) -> WorkflowResult:

        result.workflow.status = "completed"
        result.workflow.current_step = "completed"
        result.workflow.final_label = result.label

        return result

    @staticmethod
    def _mark_failed(
        result: WorkflowResult,
        step: str,
        error: Exception,
    ) -> None:

        result.workflow.status = "failed"
        result.workflow.current_step = step

        failed_result = StepResult(
            status="failed",
            reason=str(error),
        )

        if step == "spam_check":
            result.decisions.spam = failed_result

        elif step == "file_check":
            result.decisions.file = failed_result

        elif step == "questions_consecutive_check":
            result.decisions.questions_consecutive = failed_result

        elif step == "correct_check":
            result.decisions.correct = failed_result

        elif step in (
            "agent_classification",
            "special_classification",
        ):
            result.decisions.category = failed_result
