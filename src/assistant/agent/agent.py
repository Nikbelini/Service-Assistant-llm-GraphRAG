"""Цикл агента. Требования FR-7, FR-8, FR-9 и FR-10."""

import json
import re
from collections.abc import Callable

from assistant.agent.tools import TOOL_SCHEMAS, WAREHOUSES
from assistant.infra.erp_client import ErpUnavailable
from assistant.infra.grounding import _CODE, find_codes, normalize_code
from assistant.infra.llm import LLMClient, tool_message
from assistant.infra.schemas import AgentResult, ToolStep

NOT_FOUND = "В базе знаний не найдено."
ERP_RETRIES = 2
NUMBER_WORDS = {
    "один": 1, "одну": 1, "одна": 1, "одно": 1, "два": 2, "две": 2, "три": 3, "четыре": 4, "пять": 5,
    "шесть": 6, "семь": 7, "восемь": 8, "девять": 9, "десять": 10
}

_NUMBER = r"(?<![\w-])(?:\d{1,3}(?:[ \u00a0\u202f]\d{3})+|\d+)(?![\w-])"
_PRICE_OR_STOCK = re.compile(rf"({_NUMBER})\s*(?:руб|₽|шт)|остат\w*\W{{0,12}}({_NUMBER})", re.I)
SCHEMAS = {s["function"]["name"]: s["function"]["parameters"] for s in TOOL_SCHEMAS}

SYSTEM_PROMPT = f"""Ты ассистент сервисной службы по холодильному оборудованию. Отвечай по-русски, коротко.
Правила:
1. Номер детали узнавай через search_knowledge. Не придумывай номера, цены и остатки:
цену и остаток дают только get_price и check_stock.
2. Если деталь заменена, для цены, остатка и заказа используй последнюю деталь цепочки замен.
3. Склад для check_stock и create_service_order бери из вопроса. Количество для заказа бери только из вопроса.
4. create_service_order вызывай, только если пользователь просит оформить заказ.
5. Обзорные вопросы про весь корпус или линейку решай через summarize_topic.
6. Если данных нет, ответь ровно: «{NOT_FOUND}»
7. Если инструмент вернул error, не повторяй тот же вызов; расскажи об ошибке."""


def _numbers(text: str) -> set[str]:
    out = set()

    for raw in  re.findall(_NUMBER, text):
        digits = re.sub(r"\D", "", raw)
        out.add(digits)

    out.update(re.findall(r"\d+", text))
    return out


def _named_quantities(question: str) -> set[int]:
    """Количества, названные пользователем, числительные словами"""
    stripped = _CODE.sub(" ", question)
    found = {int(n) for n in re.findall(r"(?<![\w-])\d+(?![\w-])", stripped)}
    found.update(value for word, value in NUMBER_WORDS.items() if re.search(rf"(?<!\w){word}(?!\w)", stripped.lower()))
    return found


def _unsupported(answer: str, question: str, known_text: str) -> list[str]:
    """Коды, цены и остатки ответа, которых нет ни в вопросе, ни в результатах инструментов"""
    allowed_codes = set(find_codes(question)) | set(find_codes(known_text))
    bad = [code for code in find_codes(answer) if code not in allowed_codes]
    allowed_numbers = _numbers(known_text) | _numbers(question)

    for match in _PRICE_OR_STOCK.finditer(answer):
        raw = match.group(1) or match.group(2)

        if re.sub(r"\D", "", raw) not in allowed_numbers:
            bad.append(re.sub(r"\D", "", raw))

    return bad


def _summary(steps: list[ToolStep]) -> str:
    lines = []

    for step in steps:
        if step.result and "answer" in step.result:
            lines.append(step.result["answer"])
        elif step.result:
            lines.append(f"{step.tool}: {json.dumps(step.result, ensure_ascii=False)}")

    return "\n".join(lines) or "ничего выяснить не удалось"


def run_agent(
    question: str,
    llm: LLMClient,
    tools: dict[str, Callable[..., dict]],
    *,
    max_steps: int = 6,
    confirm: bool = False,
) -> AgentResult:
    """Ведёт диалог с моделью в цикле «рассуждение, вызов инструмента, наблюдение».

    Модель получает `TOOL_SCHEMAS` через `llm.chat(..., tools=...)`. Каждый вызов записывается
    в `steps` как `ToolStep`. Агент сам проверяет номер детали в аргументах (он должен встречаться
    в вопросе или в результатах прошлых инструментов), количество в заказе (должно быть названо
    пользователем) и повторные вызовы. При `ErpUnavailable` вызов повторяется до двух раз.
    Заказ создаётся только при `confirm=True`, иначе агент возвращает `needs_confirmation=True`
    и `pending_action`. Перед выдачей ответа коды, цены и остатки сверяются с вопросом
    и результатами инструментов.
    """
    # цикл, проверки аргументов, повторы, подтверждение, проверка итогового ответа
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}]
    steps: list[ToolStep] = []
    cache: dict[str, ToolStep] = {}
    tool_texts: list[str] = []
    erp_down = False

    def sources() -> list[str]:
        found: list[str] = []

        for step in steps:
            for doc in (step.result or {}).get("sources", []) if isinstance(step.result, dict) else []:
                if doc not in found:
                    found.append(doc)

        return found

    def known_codes_text() -> str:
        return "\n".join(tool_texts)

    def finish(text: str, **extra) -> AgentResult:
        return AgentResult(text, steps, sources(), **extra)

    def reject(call, args: dict, error: str) -> dict:
        steps.append(ToolStep(call.name, args, None, error))
        return {"error": error}

    while True:
        response = llm.chat(messages, tools=TOOL_SCHEMAS)
        if not response.tool_calls:
            break

        messages.append(response.as_message())
        for call in response.tool_calls:
            if len(steps) >= max_steps:
                text = f"Работа прервана: достигнут лимит в {max_steps} шагов. Успел выяснить: \n{_summary(steps)}"
                return finish(text)

            args = dict(call.arguments)
            schema = SCHEMAS.get(call.name)

            if schema is None:
                result = reject(call, args, f"неизвестный инструмент {call.name!r}; доступны: {', '.join(SCHEMAS)}")
            elif "_raw" in args:
                result = reject(
                    call, {}, "аргументы не разобраны: нужен JSON с полями " + ", ".join(schema['required']) 
                )
            elif set(args) != set(schema["required"]):
                result = reject(call, args, f"нужны ровно аргументы: {', '.join(schema['required'])}")
            else:
                for key in ("part_id", "model"):
                    code = normalize_code(args[key]) if isinstance(args.get(key), str) else None

                    if code:
                        args[key] = code

                key_text = f"{call.name}:{json.dumps(args, sort_keys=True, ensure_ascii=False)}"
                known = set(find_codes(question)) | set(find_codes(known_codes_text()))
                problem = next(
                    (
                        f"код {args[k]} не встречался ни в вопросе, ни в результатах инструментов; сначала вызови search_knowledge"
                        for k in ("part_id", "model")
                        if k in args and isinstance(args[k], str) and normalize_code(args[k]) and args[k] not in known
                    ), None
                )

                if key_text in cache:
                    old = cache[key_text]
                    steps.append(ToolStep(call.name, args, old.result, old.error))
                    result = old.result if old.result is not None else {"error": old.error}
                elif problem:
                    result = reject(call, args, problem)
                elif call.name == "create_service_order" and (
                    args["qty"] not in _named_quantities(question) or isinstance(args["qty"], bool)
                ):
                    steps.append(ToolStep(call.name, args, None, "количество не названо в вопросе"))
                    return finish("Сколько штук заказать? В запросе количество не указано, без него заказ не оформляю.")
                elif call.name == "create_service_order" and not confirm:
                    if args["warehouse"] not in WAREHOUSES:
                        result = reject(call, args, f"неизвестный склад {args['warehouse']!r}")
                    else:
                        steps.append(ToolStep(call.name, args, {"status": "pending_confirmation"}, None))
                        text = (
                            f"Нужно подтверждение: заказ {args['qty']} шт. детали {args['part_id']} на складе"
                            f"{args['warehouse']} для модели {args['model']}. Заказ не создан"
                        )
                        return finish(
                            text, needs_confirmation=True, pending_action={"tool": call.name, "args": dict(args)}
                        )
                else:
                    step = ToolStep(call.name, args)
                    for attempt in range(ERP_RETRIES + 1):
                        try:
                            step.result = tools[call.name](**args)
                            break
                        except ErpUnavailable as exc:
                            if attempt == ERP_RETRIES:
                                step.error = f"ERP недоступна после {ERP_RETRIES + 1} попыток: {exc}"
                                erp_down = True
                        except TypeError as exc:
                            step.error = f"неверные аргументы: {exc}"
                            break

                    if step.result is not None and "error" in step.result:
                        step.error = step.result["error"]

                    steps.append(step)
                    cache[key_text] = step
                    result = step.result if step.result is not None else {"error": step.error}

                    if step.result is not None:
                        tool_texts.append(json.dumps(step.result, ensure_ascii=False))

            messages.append(tool_message(call, result))

    answer = response.content.strip()
    unsupported = _unsupported(answer, question, known_codes_text())

    if unsupported:
        messages += [
            response.as_message(),
            {
                "role": "user",
                "content": f"В ответе есть данные, которых нет в вопросе и результатах инструментов: {', '.join(unsupported)}. "
                "Исправь ответ: используй только данные из вопроса и результатов инструментов."
            }
        ]

        answer = llm.chat(messages).content.strip()

        if _unsupported(answer, question, known_codes_text()):
            return finish(f"{NOT_FOUND} Проверить данные ответа по базе знаний и ERP не удалось.", refused=True)

    if not answer:
        return finish(f"Ответ сформулировать не удалось. Успел выяснить: \n{_summary(steps)}")

    if "не найдено" in answer.lower() and "базе знаний" in answer.lower():
        return finish(NOT_FOUND, refused=True)
    
    if erp_down and not ("ERP" in answer and "не удалось" in answer):
        answer += " Данные ERP получить не удалось."

    return finish(answer)
