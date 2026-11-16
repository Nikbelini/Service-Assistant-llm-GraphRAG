"""Извлечение сущностей и связей из чанка. Требование FR-1."""

import json

from assistant.infra.config import vocabulary
from assistant.infra.grounding import canonical_name, find_codes
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import (
    ENTITY_TYPES,
    EXTRACTION_SCHEMA,
    RELATION_SIGNATURES,
    Chunk,
    Entity,
    ExtractionResult,
    Relation,
)

CODE_TYPES = ("Model", "Part", "ErrorCode", "Procedure")

SYSTEM_PROMPT = """Ты извлекаешь граф знаний из фрагмента сервисной документации по холодильному оборудованию.
Верни только JSON с полями entities и relations.

Типы сущностей и как записывать имя:
- ProductLine: название линейки из словаря (Арктика)
- Model: код модели (AR-210)
- Component: название узла из словаря строчными буквами (пусковое реле)
- Part: номер детали (P-1042, P-1042B)
- ErrorCode: код ошибки (E-17)
- Procedure: номер регламента (SP-104)
- Supplier: название поставщика из словаря (КомпрессорМаш), без «ООО» и кавычек

Типы связей, направление строго от первого типа ко второму:
- BELONGS_TO: Model -> ProductLine (модель входит в линейку)
- HAS_PART: Model -> Part (в модели установлена деталь)
- IS_A: Part -> Component (деталь является узлом этого вида)
- INDICATES: ErrorCode -> Component (ошибка указывает на отказ узла)
- FIXED_BY: ErrorCode -> Procedure (ошибка устраняется по регламенту)
- REPLACED_BY: Part -> Part (старая деталь снята с производства, дальше идёт новая)
- SUPPLIED_BY: Part -> Supplier (деталь поставляет поставщик)
- FAILED: Model -> Component (в отчёте о выезде у модели зафиксирован отказ узла)

Правила:
1. Извлекай только то, что сказано в тексте. Не выдумывай коды и названия.
2. Названия линеек, узлов и поставщиков бери только из словаря:
{vocabulary}
3. Номера документов (SB-..., R-...) сущностями не являются.
4. Регламент плановых работ, где нет ошибки, детали или отказа, даёт пустые списки.
5. Коды записывай в каноническом виде: прописные латинские буквы, дефис, цифры
(ar 210 -> AR-210, Е-17 с русской Е -> E-17).

Пример. Текст: «Деталь P-1042 (пусковое реле) снята с производства, вместо неё P-1042B.
Поставщик P-1042B – ООО «ХолодКомплект»».
Ответ: {example}"""


def _example() -> str:
    def rel(a, r, b, at="Part", bt="Part"):
        return {"subject_type": at, "subject": a, "type": r, "object_type": bt, "object": b}

    return json.dumps(
        {
            "entities": [
                {"type": "Part", "name": "P-1042"},
                {"type": "Part", "name": "P-1042B"},
                {"type": "Component", "name": "пусковое реле"},
                {"type": "Supplier", "name": "ХолодКомплект"}
            ],
            "relations": [
                rel("P-1042", "REPLACED_BY", "P-1042B"),
                rel("P-1042", "IS_A", "пусковое реле", bt="Component"),
                rel("P-1042B", "SUPPLIED_BY", "ХолодКомплект", bt="Supplier")
            ],
        }, ensure_ascii=False
    )


def _parse(content: str) -> tuple[list[dict], list[dict]] | None:
    """Парсинг ответа и случае ошибки None"""
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return None

    if not isinstance(data, dict):
        return None

    entities, relations = data.get("entities"), data.get("relations")
    if not isinstance(entities, list) or not isinstance(relations, list):
        return None

    for item in entities:
        if not isinstance(item, dict) or not all(isinstance(item.get(k), str) for k in ("type", "name")):
            return None

    keys = ("subject_type", "subject", "type", "object_type", "object")
    for item in relations:
        if not isinstance(item, dict) or not all(isinstance(item.get(k), str) for k in keys):
            return None

    return entities, relations


def _confirm(entity_type: str, raw: str, chunk_codes: set[str], vocab: dict) -> Entity | None:
    if entity_type not in ENTITY_TYPES:
        return None

    name = canonical_name(entity_type, raw, vocab)
    if name is None:
        return None

    if entity_type in CODE_TYPES and name not in chunk_codes:
        return None

    return Entity(entity_type, name)


def extract_chunk(chunk: Chunk, llm: LLMClient) -> ExtractionResult:
    """Возвращает сущности и связи чанка строго по схеме графа.

    Модель вызывается через `llm.chat(..., json_schema=EXTRACTION_SCHEMA)`. Запрос описывает схему
    с направлением каждой связи и содержит хотя бы один пример. Всё, что не подтверждается текстом,
    отбрасывается и считается в `dropped`: код, которого нет в чанке, название не из словаря,
    связь с отброшенным концом. Пригодятся `find_codes` и `canonical_name` из `assistant.infra.grounding`.
    Функция не падает ни на каком ответе модели: одна повторная попытка, затем пустой результат.
    """
    # запрос к модели, разбор JSON, проверки по тексту чанка и словарю
    vocab = vocabulary()
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT.format(vocabulary=json.dumps(vocab, ensure_ascii=False), example=_example())
        },
        {"role": "user", "content": f"Идентификатор фрагмента: {chunk.chunk_id}\n\nТекст:\n{chunk.text}"}
    ]

    parsed = None
    for _ in range(2):
        parsed = _parse(llm.chat(messages, json_schema=EXTRACTION_SCHEMA, max_tokens=2048).content)
        if parsed is not None:
            break

    result = ExtractionResult(chunk.chunk_id, chunk.doc_id)
    if parsed is None:
        return result

    raw_entities, raw_relations = parsed
    chunk_codes = set(find_codes(chunk.text))
    seen: dict[str, Entity] = {}

    def keep(entity: Entity) -> None:
        seen.setdefault(entity.node_id, entity)

    for item in raw_entities:
        entity = _confirm(item["type"], item["name"], chunk_codes, vocab)
        if entity is None:
            result.dropped += 1
        else:
            keep(entity)

    relations: dict[tuple, Relation] = {}
    for item in raw_relations:
        subject = _confirm(item["subject_type"], item["subject"], chunk_codes, vocab)
        obj = _confirm(item["object_type"], item["object"], chunk_codes, vocab)
        relation = Relation(subject, item["type"], obj) if subject and obj else None

        if relation is None or item["type"] not in RELATION_SIGNATURES or not relation.matches_schema():
            result.dropped += 1
            continue

        keep(subject)
        keep(obj)
        relations.setdefault((subject.node_id, relation.type, obj.node_id), relation)

    result.entities = list(seen.values())
    result.relations = list(relations.values())
    return result
