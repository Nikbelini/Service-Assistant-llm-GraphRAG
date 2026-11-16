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
    # снять направление рёбер, разбить с учётом weight, вернуть Community с номерами по порядку
    undirected = nx.Graph()
    undirected.add_nodes_from(sorted(graph.nodes))

    for source, target, data in graph.edges(data=True):
        weight = data.get("weight", 1) + (
            undirected[source][target]["weight"] if undirected.has_edge(source, target) else 0
        )

        undirected.add_edge(source, target, weight=weight)

    parts = nx.community.louvain_communities(undirected, weight="weight", seed=SEED)
    ordered = sorted((sorted(part) for part in parts), key=lambda nodes: nodes[0])
    return [Community(number, nodes) for number, nodes in enumerate(ordered)]


def _name(graph: nx.DiGraph, node: str) -> str:
    return graph.nodes[node].get("name", node)


def _line_of(graph: nx.DiGraph, model: str) -> str:
    for target in graph.successors(model):
        if graph.edges[model, target]["type"] == "BELONGS_TO":
            return _name(graph, target)

    return "без линейки"


def community_facts(community: Community, graph: nx.DiGraph) -> str:
    """Факты и подсчёты сообщества, готовые для запроса к модели"""
    members = set(community.nodes)
    types = Counter(graph.nodes[node]["type"] for node in members)
    edges = sorted(
        ((s, t, d) for s, t, d in graph.edges(data=True) if s in members and t in members),
        key=lambda edge: (-edge[2].get("weight", 1), edge[0], edge[1])
    )

    lines = [f"Узлов: {len(members)} (" + ", ".join(f"{k}: {v}" for k, v in sorted(types.items())) + ")"]
    lines.append(f"Связей внутри сообщества: {len(edges)}")
    failures: dict[str, Counter] = defaultdict(Counter)

    for source, target, data in edges:
        if data["type"] == "FAILED":
            failures[_line_of(graph, source)][_name(graph, target)] += data.get("weight", 1)

    for line, counter in sorted(failures.items()):
        ranking = ", ".join(f"{name} - {count}" for name, count in counter.most_common())
        lines.append(f"Отказы узлов в линейке {line} (число отчётов): {ranking}")

    lines.append("Самые тяжёлые связи (вес - число документов, где связь найдена):")

    for source. target, data in edges[:MAX_EDGES_IN_PROMPT]:
        lines.append(f"{_name(graph, source)} -{data['type']}-> {_name(graph, target)} (вес {data.get('weight', 1)})")

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
