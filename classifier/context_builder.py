
from classifier.schemas import (
    InputData,
    Message,
    DialogContext,
)


def build_context(
    input_data: InputData,
    messages_before: int | None = None,
    messages_after: int | None = 1,
) -> DialogContext:
    """
    Формирует контекст диалога относительно целевого сообщения.

    messages_before:
        None — все сообщения до целевого.
        0    — не включать сообщения до целевого.
        N    — включить N ближайших сообщений до целевого.

    messages_after:
        None — все сообщения после целевого.
        0    — не включать сообщения после целевого.
        N    — включить N ближайших сообщений после целевого.

    Целевое сообщение включается всегда.
    Информация о непосредственных соседях target
    определяется независимо от размера контекста.
    """

    # Проверяем настройки контекста.
    for name, value in (
        ("messages_before", messages_before),
        ("messages_after", messages_after),
    ):
        if value is not None and (
            not isinstance(value, int) or value < 0
        ):
            raise ValueError(
                f"{name} должен быть неотрицательным целым числом "
                "или None."
            )

    messages = input_data.answer

    if not messages:
        raise ValueError(
            "Невозможно сформировать контекст: список сообщений пуст."
        )

    # Находим целевое сообщение.
    target_indices = [
        index
        for index, message in enumerate(messages)
        if message.target_key == "1"
    ]

    if len(target_indices) != 1:
        raise ValueError(
            "В диалоге должно быть ровно одно целевое сообщение "
            f"с ключом '1'. Найдено: {len(target_indices)}."
        )

    target_index = target_indices[0]
    target = messages[target_index]

    # Проверяем источник целевого сообщения.
    if target.source != "CLIENT":
        raise ValueError(
            "Целевое сообщение должно принадлежать клиенту. "
            f"Получен источник: {target.source}."
        )

    # Проверяем принадлежность сообщений одной чат-сессии.
    session_id = target.session_id

    for message in messages:
        if message.session_id != session_id:
            raise ValueError(
                "Обнаружены сообщения из разных чат-сессий."
            )

    # Проверяем непосредственных соседей target
    # в исходном диалоге, не учитывая настройки окна контекста.
    has_previous_client_message = (
        target_index > 0
        and messages[target_index - 1].source == "CLIENT"
    )

    has_next_client_message = (
        target_index + 1 < len(messages)
        and messages[target_index + 1].source == "CLIENT"
    )

    def convert_message(raw_message) -> Message:
        if not raw_message.request_id:
            raise ValueError(
                "У сообщения отсутствует ID запроса."
            )

        return Message(
            message_id=raw_message.request_id,
            text=raw_message.text,
            source=raw_message.source,
            timestamp=raw_message.timestamp,
            scenario=raw_message.scenario,
            agent=raw_message.agent,
        )

    # Определяем границы истории до целевого сообщения.
    if messages_before is None:
        before_start = 0
    else:
        before_start = max(
            0,
            target_index - messages_before,
        )

    # Определяем границы истории после целевого сообщения.
    if messages_after is None:
        after_end = len(messages)
    else:
        after_end = min(
            len(messages),
            target_index + 1 + messages_after,
        )

    # Формируем обе части истории.
    history_before_target = [
        convert_message(message)
        for message in messages[before_start:target_index]
    ]

    history_after_target = [
        convert_message(message)
        for message in messages[target_index + 1:after_end]
    ]

    return DialogContext(
        session_id=session_id,
        history_before_target=history_before_target,
        target_message=convert_message(target),
        history_after_target=history_after_target,
        has_previous_client_message=has_previous_client_message,
        has_next_client_message=has_next_client_message,
    )
