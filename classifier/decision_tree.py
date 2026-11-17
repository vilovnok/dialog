
from classifier.categories import (
    CORRECT_CATEGORY,
    is_valid_category,
)

from classifier.schemas import (
    DialogContext,
    DecisionResult,
    CategoryResult,
    StepResult,
    WorkflowResult,
)

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

            # print(f"Context перед spam check: \n\n{context}\n***\n")

            spam_result = self.model.check_spam(context)

            result.decisions.spam = StepResult(
                status="completed",
                label=spam_result.decision,
                reason=spam_result.reason,
            )

            if spam_result.decision == "spam":

                result.spam = True
                result.workflow.spam = True

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
            # 2. CORRECT CHECK
            # --------------------------------------------------

            active_step = "correct_check"
            result.workflow.current_step = active_step

            correct_result = self.model.check_correct(context)

            result.decisions.correct = StepResult(
                status="completed",
                label=correct_result.decision,
                reason=correct_result.reason,
            )

            if correct_result.decision == "correct":

                result.label = CORRECT_CATEGORY

                result.decisions.category = StepResult(
                    status="skipped",
                    reason="Переспрос признан обоснованным.",
                )

                return self._finish(result)

            # --------------------------------------------------
            # 3. CATEGORY CLASSIFICATION
            # --------------------------------------------------

            active_step = "category_classification"
            result.workflow.current_step = active_step

            category_result = self.model.classify_category(context)

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
    # VALIDATION
    # --------------------------------------------------

    @staticmethod
    def _validate_category(
        category: CategoryResult,
    ) -> None:

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

        elif step == "correct_check":
            result.decisions.correct = failed_result

        elif step == "category_classification":
            result.decisions.category = failed_result
