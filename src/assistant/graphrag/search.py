"""Два режима поиска по графу. Требования FR-4, FR-5 и FR-10."""

import json
import re
from itertools import combinations

import networkx as nx

from assistant.infra.grounding import code_type, find_codes, find_names
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import MAP_SCHEMA, Answer, KnowledgeBase

NOT_FOUND = "В базе знаний не найдено"
MAX_FACTS_CHARS = 8000

MAX_PATH_LEN = 5
MAX_PATHS_PER_PAIR = 100
MAX_REPLACEMENT_DEPTH = 12
NEIGHBOURHOOD_DEPTH = 2


def _doc(chunk_id: str) -> str:
    return chunk_id.split("#", 1)[0]


def _stem_match(name: str, lowered: str) -> bool:
    """Название узла в любой падежной форме: каждое слово встречается по основе"""
    words = name.lower().split()
    return all(word in lowered or (len(word) > 4 and word[:-2] in lowered) for word in words)


def _anchors(question: str, kb: KnowledgeBase) -> tuple[list[str], bool]:
    """Опорные узлы вопроса и признак того, что какой-то код в графе отсутствует"""
    anchors: list[str] = []

    for code in find_codes(question):
        node = f"{code_type(code)}:{code}"

        if node not in kb.graph:
            return [], True

        if node not in anchors:
            anchors.append(node)


    lowered = question.lower().replace("ё", "е")
    names = {(kind, name) for kind, name in find_names(question, kb.vocabulary)}

    for kind, values in kb.vocabulary.items():
        names.update((kind, name) for name in values if _stem_match(name.lower().replace("ё", "е"), lowered))

    for kind, name in sorted(names):
        node = f"{kind}:{name}"

        if node in kb.graph and node not in anchors:
            anchors.append(node)

    return anchors, False


def _add_real_edges(graph: nx.DiGraph, edges: dict[tuple[str, str], int], a: str, b: str, rank: int) -> None:
    """Добавляет только реально существующие направленные рёбра"""
    if graph.has_edges(a, b):
        edges[(a, b)] = min(edges.get((a, b), rank), rank)

    if graph.has_edges(b, a):
        edges[(b, a)] = min(edges.get((b, a), rank), rank)


def _replacement_edges(graph: nx.DiGraph, edges: dict[tuple[str, str], int], parts: set[str]) -> None:
    """Добавляет полные цепочки REPLACED_BY, включая ветвления, без зацикливания"""
    for start in sorted(parts):
        queue: list[tuple[str, int, frozenset[str]]] = [(start, 0, frozenset({start}))]

        while queue:
            current, depth, visited = queue.pop(0)

            if depth >= MAX_REPLACEMENT_DEPTH:
                continue

            successors = sorted(
                (
                    target for target in graph.successors(current)
                    if graph.edges[current, target].get("type") == "REPLACED_BY"
                ),
                key=lambda target: (-graph.edges[current, target].get("weight", 1), target)
            )

            for target in successors:
                if target in visited:
                    continue

                edges[(current, target)] = min(edges.get((current, target), 0), 0)
                queue.append((target, depth + 1, visited | {target}))


def _edge_ids(graph: nx.DiGraph, undirected: nx.Graph, anchors: list[str]) -> list[tuple[str, str]]:
    """Рёбра на путях между опорными узлами либо в окрестности единственного узла"""
    edges: dict[tuple[str, str], int] = {}

    if len(anchors) >= 2:
        for first, second in combinations(anchors, 2):
            if first not in undirected or second not in undirected:
                continue

            if not nx.has_path(undirected, first, second):
                continue

            # Ищем короткие простые пути, но не разрешаем обход через третью
            paths = nx.all_simple_paths(
                undirected, first, second, cutoff=MAX_PATH_LEN
            )

            accepted = 0
            for path in paths:
                intermediate = path[1:-1]

                if any(graph.nodes[node].get("type") in {"Model", "ProductLine"} for node in intermediate):
                    continue

                rank = len(path) - 1
                for a, b in zip(path, path[1:]):
                    _add_real_edges(graph, edges, a, b, rank)

                accepted += 1
                if accepted >= MAX_PATHS_PER_PAIR:
                    break
    elif anchors:
        frontier = {anchors[0]}
        seen = set(frontier)

        for depth in range(NEIGHBOURHOOD_DEPTH):
            following: set[str] = set()

            for node in sorted(frontier):
                for other in sorted(undirected.neighbors(node)):
                    _add_real_edges(graph, edges, node, other, depth)

                    if other not in seen:
                        following.add(other)

            seen.update(following)
            frontier = following

    parts = {node for pair in edges for node in pair if graph.nodes[node]["type"] == "Part"}
    _replacement_edges(graph, edges, parts)

    return sorted(edges, key=lambda pair: (edges[pair], -graph.edges[pair].get("weight", 1), pair[0],  pair[1]))


def _fact(graph: nx.DiGraph, source: str, target: str) -> str:
    data = graph.edges[source, target]
    text = f"{graph.nodes[source].get('name', source)} -{data['type']}-> {graph.nodes[target].get('name', source)}"
    return text + (f" (найдено в {data['weight']} документах)" if data.get("weight", 1) > 1 else "")


def local_search(question: str, kb: KnowledgeBase, llm: LLMClient) -> Answer:
    """Отвечает на вопрос о конкретных сущностях по фактам из графа.

    Опорные сущности ищутся без модели: коды через `find_codes`, названия через `find_names`.
    При двух и более опорных сущностях берутся факты на путях длиной до трёх связей между ними
    и цепочки `REPLACED_BY` от деталей на этих путях. При одной берётся окрестность на глубину
    до двух связей. Модель получает пронумерованный список фактов и отвечает только по нему.
    Если в вопросе есть код, которого нет в графе, или фактов нет, возвращается
    `Answer(..., refused=True)` без вызова модели.
    """
    # опорные сущности, отбор фактов, запрос к модели, источники
    anchors, unknown_code = _anchors(question, kb)

    if unknown_code or not anchors:
        return Answer(NOT_FOUND, refused=True)

    graph = kb.graph
    pairs = _edge_ids(graph, graph.to_undirected(as_view=True), anchors)
    facts: list[str] = []
    used: list[tuple[str, str]] = []
    size = 0

    for source, target in pairs:
        fact = _fact(graph, source, target)
        line = f"{len(facts) + 1}. {fact}"   
        if size + len(line) + 1 > MAX_FACTS_CHARS:
            continue
        
        facts.append(line)
        used.append((source, target))
        size += len(line) + 1

    if not facts:
        return Answer(NOT_FOUND, refused=True)

    messages = [
        {
            "role": "system",
            "content": "Ты отвечаешь на вопросы сервисной службы только по списку фактов. Не добавляй сведения, которых нет в фактах. "
                "Если есть цепочка REPLACED_BY, укажи последнюю деталь этой цепочки. Не путай детали разных моделей. Коды переписывай точно. "
                f"Если ответа в фактах нет, напиши ровно: «{NOT_FOUND}». Отвечай кратко и прямо на вопрос."
        },
        {"role": "user", "content": "Факты графа:\n" + "\n".join(facts) + f"\n\nВопрос: {question}"}
    ]

    text = llm.chat(messages).content.strip()
    sources = sorted({_doc(chunk) for pair in used for chunk in graph.edges[pair].get("sources", [])})
    refused = not text or ("не найдено" in text.lower() and "базе знаний" in text.lower())
    return Answer(text or NOT_FOUND, sources, [line.split(". ", 1)[1] for line in facts], refused)


def _parse_partial(content: str) -> tuple[str, int] | None:
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        match = re.search(r"\{.*\}", content or "", re.S)

        try:
            data = json.loads(match.group(0)) if match else None
        except json.JSONDecodeError:
            return None

    if not isinstance(data, dict) or not isinstance(data.get("answer"), str):
        return None

    score = data.get("score")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        return None

    return data["answer"].strip(), max(0, min(10, int(score)))


def global_search(question: str, kb: KnowledgeBase, llm: LLMClient) -> Answer:
    """Отвечает на обзорный вопрос по сводкам сообществ в два шага.

    Шаг map: для каждой сводки модель даёт частичный ответ и оценку полезности от 0 до 10.
    Шаг reduce: частичные ответы с оценкой выше 0 объединяются в итоговый ответ.
    Если таких ответов нет, возвращается `Answer(..., refused=True)`.
    """
    partials: list[tuple[int, str, object]] = []

    for community in kb.communities:
        messages = [
            {
                "role": "system",
                "content": "Тебе дана сводка части графа знаний сервисной службы и вопрос. Ответь на вопрос по сводке "
                'и оцени полезность сводки для ответа от 0 до 10. Верни JSON {"answer": "...", "score": N}. '
                "score от 0 до 10. Если сведений для ответа нет, score=0.",
            },
            {"role": "user", "content": f"Сводка:\n{community.summary}\n\nВопрос: {question}"}
        ]

        parsed = _parse_partial(llm.chat(messages, json_schema=MAP_SCHEMA, max_tokens=512).content)
        if parsed and parsed[1] > 0 and parsed[0]:
            partials.append((parsed[1], parsed[0], community))

    if not partials:
        return Answer(NOT_FOUND, refused=True)

    partials.sort(key=lambda item: (-item[0], item[1]))
    joined = "\n".join(f"{i}. (оценка {score}) {answer}" for i, (score, answer, _) in enumerate(partials, start=1))

    messages = [
        {
            "role": "system",
            "content": "Объедини частичные ответы в один итоговый ответ на вопрос. Используй только их содержание, "
            "Не складывай и не усредняй числа самостоятельно. Коды и числа не изменяй."
            "Если частичные ответы противоречат друг другу, укажи, что данных недостаточно."
        },
        {"role": "user", "content": f"Частичные ответы:\n{joined}\n\nВопрос: {question}"}
    ]

    text = llm.chat(messages).content.strip()
    docs = sorted(
        {
            _doc(chunk)
            for _, _, community in partials
            for node in community.nodes
            for chunk in kb.graph.nodes[node].get("sources", [])
        }
    )

    refused = not text or ("не найдено" in text.lower() and "базе знаний" in text.lower())
    return Answer(text or NOT_FOUND, docs, [answer for _, answer, _ in partials], refused)
