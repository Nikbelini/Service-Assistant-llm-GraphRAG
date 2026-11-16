"""Сборка графа знаний из результатов извлечения. Требование FR-2."""

import networkx as nx

from assistant.infra.schemas import ENTITY_TYPES, ExtractionResult


def build_graph(results: list[ExtractionResult]) -> nx.DiGraph:
    """Объединяет результаты по всем чанкам в один направленный граф.

    Узел: идентификатор `Entity.node_id`, атрибуты `type`, `name`, `sources` (список chunk_id).
    Ребро: атрибуты `type`, `weight` (число разных чанков, где связь найдена), `sources`.
    Связи, не соответствующие схеме (`Relation.matches_schema`), отбрасываются. Если между двумя
    деталями есть `REPLACED_BY` в обе стороны, остаётся связь с большим весом, при равенстве
    отбрасываются обе. Число таких конфликтов записывается в `graph.graph["conflicts"]`.
    """
    # узлы, рёбра с весами, проверка схемы, разбор встречных замен
    node_sources: dict[str, set[str]] = {}
    node_info: dict[str, tuple[str, str]] = {}
    edge_sources: dict[tuple[str, str], tuple[str, set[str]]] = {}


    def add_node(entity, chunk_id: str) -> None:
        node_info[entity.node_id] = (entity.type, entity.name)
        node_sources.setdefault(entity.node_id, set()).add(chunk_id)


    for result in results:
        for entity in result.entities:
            if entity.type in ENTITY_TYPES:
                add_node(entity, result.chunk_id)

        for relation in result.relations:
            if not relation.matches_schema():
                continue

            add_node(relation.subject, result.chunk_id)
            add_node(relation.object, result.chunk_id)
            key = (relation.subject.node_id, relation.object.node_id)
            edge_sources.setdefault(key, (relation.type, set()))[1].add(result.chunk_id)

    graph = nx.DiGraph(conflicts=0)

    for node, (node_type, name) in node_info.items():
        graph.add_node(node, type=node_type, name=name, sources=sorted(node_sources[node]))

    for (source, target), (kind, chunks) in edge_sources.items():
        graph.add_edge(source, target, type=kind, weight=len(chunks), sources=sorted(chunks))

    conflicts = 0

    for source, target in list(graph.edges):
        if not (graph.has_edge(source, target) and graph.has_edge(target, source)):
            continue

        forward, backward = graph.edges[source, target], graph.edges[target, source]

        if forward["type"] != "REPLACED_BY" or backward["type"] != "REPLACED_BY":
            continue

        conflicts += 1

        if forward["weight"] > backward["weight"]:
            graph.remove_edge(target, source)
        elif forward["weight"] < backward["weight"]:
            graph.remove_edge(source, target)
        else:
            graph.remove_edge(source, target)
            graph.remove_edge(target, source)

    graph.graph["conflicts"] = conflicts

    return graph