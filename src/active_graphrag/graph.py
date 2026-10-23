import json
import re
import networkx as nx
import numpy as np
import yaml


def parse_triples(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        records = json.loads(text)
    except (ValueError, TypeError):
        return []
    if isinstance(records, dict):
        records = records.get("triples", [])
    if not isinstance(records, list):
        return []
    triples = []
    for row in records:
        if isinstance(row, dict):
            row = [row.get(k) for k in ("head", "relation", "tail")]
        if not isinstance(row, (list, tuple)) or len(row) != 3:
            continue
        if not all(isinstance(x, str) and x.strip() for x in row):
            continue
        h, r, t = [x.strip() for x in row]
        triple = (h, " ".join(r.lower().split()), t)
        if triple not in triples:
            triples.append(triple)
    return triples


class GraphMemory:
    def __init__(self, embedder, strong=0.85, weak=0.65):
        self.graph = nx.MultiDiGraph()
        self.embedder, self.strong, self.weak = embedder, strong, weak
        self.serial = 0

    def fuse(self, triples):
        labels = list(dict.fromkeys(x for h, _, t in triples for x in (h, t)))
        if not labels:
            return []
        old = sorted(self.graph.nodes)
        new_vectors = self.embedder.encode(labels)
        scores = new_vectors @ self.embedder.encode([self.graph.nodes[n]["label"] for n in old]).T if old else None
        mapping, audit = {}, []
        for i, label in enumerate(labels):
            best = old[int(np.argmax(scores[i]))] if old else None
            score = float(np.max(scores[i])) if old else None
            if best is not None and score >= self.strong:
                node = best
                decision = "merge"
            else:
                node = f"n{self.serial:08d}"
                self.serial += 1
                aliases = [best] if best is not None and score >= self.weak else []
                self.graph.add_node(node, label=label, alias_candidates=aliases)
                decision = "alias" if aliases else "new"
            mapping[label] = node
            audit.append(dict(label=label, target=best, similarity=score, decision=decision, retained=node))
        for h, r, t in triples:
            self.graph.add_edge(mapping[h], mapping[t], key=r, relation=r)
        return audit

    def prune(self, node):
        if node not in self.graph:
            raise ValueError("Pruning target does not exist")
        removed = [(u, d["relation"], v) for u, v, d in self.graph.edges(data=True) if u == node or v == node]
        self.graph.remove_node(node)
        for _, attrs in self.graph.nodes(data=True):
            attrs["alias_candidates"] = [n for n in attrs["alias_candidates"] if n != node]
        return removed

    def snapshot(self):
        return {"nodes": [{"id": n, "label": d["label"], "outgoing": [
            {"relation": attrs["relation"], "target": target}
            for _, target, attrs in self.graph.out_edges(n, data=True)],
            "alias_candidates": list(d["alias_candidates"])} for n, d in self.graph.nodes(data=True)]}

    def serialize(self, format="yaml"):
        return serialize(self.snapshot(), format)


def serialize(snapshot, format="yaml"):
    if format == "yaml":
        return yaml.safe_dump(snapshot, sort_keys=False, allow_unicode=True)
    if format == "json":
        return json.dumps(snapshot, ensure_ascii=False, indent=2)
    if format == "triples":
        lines = []
        for node in snapshot["nodes"]:
            lines.append(json.dumps({k: v for k, v in node.items() if k != "outgoing"}, ensure_ascii=False))
            lines.extend(json.dumps([node["id"], e["relation"], e["target"]], ensure_ascii=False) for e in node["outgoing"])
        return "\n".join(lines)
    raise ValueError(format)
