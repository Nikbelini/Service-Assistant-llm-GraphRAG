"""Сообщества графа и их сводки. Требование FR-3."""

from collections import Counter, defaultdict

import networkx as nx

from assistant.infra.llm import LLMClient
from assistant.infra.schemas import Community

SEED = 42
MAX_WORDS = 150
MAX_EDGES_IN_PROMPT = 40


def detect_communities(graph: nx.DiGraph) -> list[Community]:
    """Разбивает узлы на сообщества алгоритмом Лувена с фиксированным `seed`.

    Каждый узел входит ровно в одно сообщество. Поле `summary` здесь остаётся пустым.
    """
    if graph.number_of_nodes() == 0:
        return []
    
    # снять направление рёбер, разбить с учётом weight, вернуть Community с номерами по порядку
    undirected = nx.Graph()
    undirected.add_nodes_from(sorted(graph.nodes))

    for source, target, data in graph.edges(data=True):
        weight = max(1, data.get("weight", 1))
        if undirected.has_edge(source, target):
            undirected[source][target]["weight"] += weight
        else:
            undirected.add_edge(source, target, weight=weight)

    # Louvain требует рёбра, изолированные узлы остаются отдельными группами
    parts = nx.community.louvain_communities(undirected, weight="weight", seed=SEED)
    ordered = sorted((sorted(part) for part in parts), key=lambda nodes: nodes[0])
    return [Community(number, nodes) for number, nodes in enumerate(ordered)]


def _name(graph: nx.DiGraph, node: str) -> str:
    return graph.nodes[node].get("name", node)


def _line_of(graph: nx.DiGraph, model: str) -> str:
    for target in graph.successors(model):
        if graph.edges[model, target].get("type") == "BELONGS_TO":
            return _name(graph, target)

    return "без линейки"


def failure_statistics(graph: nx.DiGraph) -> dict[str, Counter]:
    """Отказы по всему графу. Считается по рёбрам FAILED от моделей. Сообщества сюда не входят"""
    failures: dict[str, Counter] = defaultdict(Counter)
    for source, target, data in graph.edges(data=True):
        if data.get("type") != "FAILED":
            continue

        if graph.nodes[source].get("type") != "Model":
            continue

        line = _line_of(graph, source)
        component = _name(graph, target)
        failures[line][component] += data.get("weight", 1)

    return failures


def _failure_lines(graph: nx.DiGraph) -> list[str]:
    lines: list[str] = []

    for line, counter in sorted(failure_statistics(graph).items()):
        ranking = ", ".join(
            f"{name} — {count}" for name, count in counter.most_common()
        )

        lines.append(f"Отказы по линейке {line} (число отчётов): {ranking}")

    return lines


def community_facts(community: Community, graph: nx.DiGraph) -> str:
    """Факты и подсчёты сообщества, готовые для запроса к модели"""
    members = set(community.nodes)
    types = Counter(graph.nodes[node]["type"] for node in members)
    edges = sorted(
        (
            (source, target, data)
            for source, target, data in graph.edges(data=True) 
            if source in members and target in members
        ), 
        key=lambda edge: (-edge[2].get("weight", 1), edge[0], edge[1]),
    )

    lines = [
        f"Узлов: {len(members)} ("
        + ", ".join(f"{kind}: {count}" for kind, count in sorted(types.items()))
        + ")",
        f"Связей внутри сообщества: {len(edges)}",
        "Глобальная статистика отказов по всему графу (не ограничена сообществом):",
    ]
        
    stats = _failure_lines(graph)
    lines.extend(stats or ["Связи FAILED в графе отсутствуют."])
    lines.append("Факты внутри сообщества:")
    
    for source, target, data in edges[:MAX_EDGES_IN_PROMPT]:
        lines.append(
            f"{_name(graph, source)} -{data['type']}-> {_name(graph, target)} "
            f"(вес {data.get('weight', 1)})"
        )
    
    return "\n".join(lines)


def summarize_community(community: Community, graph: nx.DiGraph, llm: LLMClient) -> str:
    """Сводка сообщества не длиннее 150 слов: какие сущности входят, как связаны, что встречается чаще.

    Все числа считает код по весам рёбер и передаёт модели готовыми. Модель их только формулирует.
    """
    # собрать факты и подсчёты сообщества, попросить модель написать сводку
    messages = [
        {
            "role": "system",
            "content": "Ты пишешь краткие сводки по графу знаний сервисной службы. Пиши по-русски, не более 120 слов. "
            "Опиши, какие сущности входят в сообщество, как они связаны и что встречается чаще всего. "
            "Используй только приведённые факты и числа, свои числа не считай и ничего не добавляй."
        },
        {"role": "user", "content": community_facts(community, graph)}
    ]

    words = llm.chat(messages, max_tokens=400).content.split()
    return " ".join(words[:MAX_WORDS])
