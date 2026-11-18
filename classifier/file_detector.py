
from classifier.schemas import DialogContext


def check_file(context: DialogContext) -> bool:
    """
    Проверяет, относится ли обращение к категории «Файл».

    Условия:
    1. Текст целевого сообщения клиента равен "-".
    2. Непосредственно следующее сообщение принадлежит CLIENT
       и его текст равен "-".

    Возвращает:
        True  — категория «Файл».
        False — категория «Файл» не определена.
    """

    if context.target_message.text == "-":
        return True

    if not context.history_after_target:
        return False

    next_message = context.history_after_target[0]

    return (
        next_message.source == "CLIENT"
        and next_message.text == "-"
    )
