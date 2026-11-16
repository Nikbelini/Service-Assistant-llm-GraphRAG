"""Инструменты агента. Требование FR-6."""

from collections.abc import Callable

from assistant.graphrag.search import global_search, local_search
from assistant.infra.grounding import code_type, normalize_code
from assistant.infra.erp_client import ErpClient, ErpNotFound, ErpRejected
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import KnowledgeBase

# Описания пяти инструментов в формате function calling:
# search_knowledge, summarize_topic, check_stock, get_price, create_service_order.
# заполните список. У параметра warehouse четыре склада задаются перечислением.
WAREHOUSES = ["Москва", "Санкт-Петербург", "Казань", "Екатеринбург"]


def _tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }

_QUESTION = {"type": "string", "description": "Вопрос на русском языке со всеми кодами из задачи"}
_PART = {"type": "string", "description": "Номер детали, например P-1042 или P-1042B"}
_WAREHOUSE = {"type": "string", "enum": WAREHOUSES, "description": "Склад"}

TOOL_SCHEMAS: list[dict] = [
    _tool(
        "search_knowledge",
        "Поиск по графу знаний: какой узел отказал, какая деталь стоит в модели, чем она заменена, кто поставщик, "
        "по какому регламенту устранять ошибку. Здесь узнают номер детали. Цен и остатков здесь нет.",
        {"question": _QUESTION},
        ["question"],
    ),
    _tool(
        "summarize_topic",
        "Обзорный вопрос по всей базе знаний: что чаще всего отказывает, какие тенденции в линейке.",
        {"question": _QUESTION},
        ["question"],
    ),
    _tool(
        "check_stock",
        "Остаток детали на складе в ERP и признак снятия с производства.",
        {"part_id": _PART, "warehouse": _WAREHOUSE},
        ["part_id", "warehouse"],
    ),
    _tool(
        "get_price", 
        "Цена детали в рублях в ERP и признак снятия с производства.", 
        {"part_id": _PART}, ["part_id"]
    ),
    _tool(
        "create_service_order",
        "Создать сервисный заказ на деталь. Выполняется только после подтверждения человеком.",
        {
            "part_id": _PART,
            "warehouse": _WAREHOUSE,
            "qty": {"type": "integer", "minimum": 1, "description": "Количество, названное пользователем"},
            "model": {"type": "string", "description": "Код модели, например AR-210"},
        },
        ["part_id", "warehouse", "qty", "model"],
    )
]


def _part(value) -> tuple[str | None, str | None]:
    code = normalize_code(value) if isinstance(value, str) else None

    if code is None or code_type(code) != "Part":
        return None, f"некорректный номер детали {value!r}: нужен вид P-1042 или P-1042B"

    return code, None


def _warehouse(value) -> str | None:
    return None if value in WAREHOUSES else f"неизвестный склад {value!r}: допустимы {', '.join(WAREHOUSES)}"


def _erp(call: Callable[[], dict]) -> dict:
    """Различимые исходы. Если данных нет, то передаётся агенту"""
    try:
        return call()
    except ErpNotFound as exc:
        return {"error": f"данных нет: ERP не знает такой записи ({exc})"}
    except ErpRejected as exc:
        return {"error": f"ERP отклонила запрос: {exc}"}


def build_tools(kb: KnowledgeBase, llm: LLMClient, erp: ErpClient) -> dict[str, Callable[..., dict]]:
    """Словарь «имя инструмента -> функция». Каждая функция возвращает словарь.

    Функция проверяет свои аргументы до обращения к ERP: склад из списка, номер детали вида P-1042
    или P-1042B, количество больше нуля. При ошибке возвращает {"error": "..."} и ничего не выполняет.
    `ErpNotFound` и `ErpRejected` превращаются в {"error": "..."} с разным текстом, чтобы исходы
    оставались различимыми. `ErpUnavailable` передаётся дальше: повторы делает агент (FR-8).
    """
    # пять функций поверх local_search, global_search и клиента ERP

    def search_knowledge(question: str) -> dict:
        answer = local_search(str(question), kb, llm)
        return {"answer": answer.text, "facts": answer.facts, "sources": answer.sources, "refused": answer.refused}


    def summarize_topic(question: str) -> dict:
        answer = global_search(str(question), kb, llm)
        return {"answer": answer.text, "sources": answer.sources, "refused": answer.refused}


    def check_stock(part_id: str, warehouse: str) -> dict:
        code, error = _part(part_id)
        error = error or _warehouse(warehouse)

        if error:
            return {"error": error}

        return _erp(lambda: erp.get_price(code, warehouse))


    def get_price(part_id: str) -> dict:
        code, error = _part(part_id)

        if error:
            return {"error": error}

        return _erp(lambda: erp.get_price(code))


    def create_service_order(part_id: str, warehouse: str, qty: int, model: str) -> dict:
        code, error = _part(part_id)
        model_code = normalize_code(model) if isinstance(model, str) else None

        if not error and (model_code is None or code_type(model_code) != "Model"):
            error = f"некорректный код модели {model!r}: нужен вид AR-210"

        error = error or _warehouse(warehouse)

        if not error and (isinstance(qty, bool) or not isinstance(qty, int) or qty <= 0):
            error = f"количество должно быть целым числом больше нуля, получено {qty!r}"

        if error:
            return {"error": error}

        return _erp(lambda: erp.create_order(code, warehouse, qty, model_code))

    return {
        "search_knowledge": search_knowledge,
        "summarize_topic": summarize_topic,
        "check_stock": check_stock,
        "get_price": get_price,
        "create_service_order": create_service_order
    }