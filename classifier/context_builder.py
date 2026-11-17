from classifier.schemas import (
    InputData,
    Message,
    DialogContext,
)


def build_context(input_data: InputData) -> DialogContext:
    """
    Формирует контекст диалога для классификации.

    Контекст:
    1. Все сообщения до целевого сообщения клиента.
    2. Целевое сообщение клиента.
    3. Непосредственно следующее сообщение сотрудника.

    Все последующие сообщения игнорируются.
    """

    messages = input_data.answer

    if len(messages) < 2:
        raise ValueError(
            "Для формирования контекста необходимо "
            "минимум два сообщения."
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

    # Следующее сообщение должно непосредственно следовать за целевым.
    if target_index + 1 >= len(messages):
        raise ValueError(
            "После целевого сообщения отсутствует сообщение агента."
        )

    followup = messages[target_index + 1]

    if followup.source != "EMPLOYEE":
        raise ValueError(
            "Следующее после целевого сообщения должно принадлежать "
            f"сотруднику. Получен источник: {followup.source}."
        )

    # Проверяем принадлежность сообщений одной чат-сессии.
    session_id = target.session_id

    for message in messages:
        if message.session_id != session_id:
            raise ValueError(
                "Обнаружены сообщения из разных чат-сессий."
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

    # В историю включаем только сообщения до целевого.
    history_before_target = [
        convert_message(message)
        for message in messages[:target_index]
    ]

    return DialogContext(
        session_id=session_id,
        history_before_target=history_before_target,
        target_message=convert_message(target),
        agent_followup=convert_message(followup),
    )