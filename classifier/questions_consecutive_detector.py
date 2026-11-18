
from classifier.schemas import DialogContext


def has_consecutive_client_messages(
    context: DialogContext,
) -> bool:
    """
    Проверяет, принадлежит ли непосредственно следующее
    после целевого сообщения сообщение клиенту.

    Условия:
    1. После целевого сообщения существует хотя бы одно сообщение.
    2. Первое сообщение после целевого принадлежит CLIENT.

    Возвращает:
        True  — структурное условие выполнено.
        False — структурное условие не выполнено.
    """

    if not context.history_after_target:
        return False

    next_message = context.history_after_target[0]

    return next_message.source == "CLIENT"
