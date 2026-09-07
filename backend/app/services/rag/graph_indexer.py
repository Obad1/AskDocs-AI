"""Cross-document relation graph indexer (networkx).

Builds a bipartite-ish graph of documents <-> extracted entities and exposes
local neighborhood queries. Pure local; entities are supplied by the caller
(extraction is owned by the ingestion/parsing pipeline, not this module).

Contract:
  add_document(doc_id, text, entities?) -> registers doc node + entity edges
  add_relation(a, b, relation?)          -> explicit edge between any two nodes
  query(doc_id, max_hops)                -> ego-graph {nodes, edges}
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import networkx as nx


class GraphIndexer:
    def __init__(self) -> None:
        self.graph = nx.Graph()
        self.doc_texts: Dict[str, str] = {}

    def add_document(
        self, doc_id: str, text: str, entities: Optional[List[str]] = None
    ) -> None:
        self.doc_texts[doc_id] = text
        self.graph.add_node(doc_id, type="document")
        for e in entities or []:
            if not self.graph.has_node(e):
                self.graph.add_node(e, type="entity")
            self.graph.add_edge(doc_id, e, relation="mentions")

    def add_relation(self, a: str, b: str, relation: str = "related") -> None:
        self.graph.add_edge(a, b, relation=relation)

    def query(self, doc_id: str, max_hops: int = 2) -> Dict[str, Any]:
        if doc_id not in self.graph:
            return {"nodes": [], "edges": []}
        nodes = nx.ego_graph(self.graph, doc_id, radius=max_hops)
        subgraph = self.graph.subgraph(nodes)
        return {
            "nodes": [
                {"id": n, **(self.graph.nodes[n])} for n in nodes
            ],
            "edges": [
                {"source": u, "target": v, **(d)}
                for u, v, d in subgraph.edges(data=True)
            ],
        }
