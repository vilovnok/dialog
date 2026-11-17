
AGENT_CATEGORIES = {
    "cb_gigachat_01_right_to_sign_supervisor",
    "cb_gigachat_02_loan_MMB",
    "cb_gigachat_03_self_encashment_service",
    "cb_gigachat_05_business_credit_card",
    "cb_gigachat_06_salary_project",
    "cb_gigachat_07_acquiring",
    "cb_gigachat_08_sberbusiness",
    "cb_gigachat_09_statement",
    "cb_gigachat_10_payments",
    "cb_gigachat_11_business_card_commission",
    "cb_gigachat_11_business_card_transactions",
    "cb_gigachat_12_bank_account_restrictions",
    "cb_gigachat_13_main_deposits",
    "cb_gigachat_14_references",
    "cb_gigachat_15_account_restrictions",
    "cb_gigachat_16_service_packages",
    "cb_gigachat_19_rag_dtb_tax_digital_signature",
    "cb_gigachat_20_collection",
    "cb_gigachat_21_business_card_limits",
    "cb_gigachat_22_rag_fast_payment",
    "cb_gigachat_27_rid_kv_edo_fl",
    "cb_gigachat_28_cash_transactions_of_legal_entities",
    "cb_gigachat_30_VED",
    "cb_gigachat_32_legal_case_changes",
    "cb_gigachat_33_letter_of_credit",
    "cb_gigachat_38_manager_ved",
    "cb_gigachat_39_cifrovoyrubl",
}


SPECIAL_CATEGORIES = {
    "ЗДА",
    "Комплаенс",
    "Налоги",
    "Программа лояльности Сбербизнес Спасибо",
    "Продукты физлиц",
    "Сберрейтинг",
    "СБК и Сберказначейство",
    "Сервисы Коруса",
    "Счет-фактуры",
    "Файл",
    "Вопросы подряд",
}


OTHER_CATEGORY = "Другое"
CORRECT_CATEGORY = "correct"


ALL_CATEGORIES = (
    AGENT_CATEGORIES
    | SPECIAL_CATEGORIES
    | {OTHER_CATEGORY}
)


def is_valid_category(
    category_type: str,
    label: str,
) -> bool:

    if category_type == "agent":
        return label in AGENT_CATEGORIES

    if category_type == "special":
        return label in SPECIAL_CATEGORIES

    if category_type == "other":
        return label == OTHER_CATEGORY

    return False
