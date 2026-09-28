"""Shared retrieval implementation for the final RepoEval study.

The graph used here is deliberately named a co-import similarity graph: two
chunks are connected when they mention the same imported top-level module.  It
is not a directed repository dependency graph.
"""

from __future__ import annotations

import ast
import hashlib
import math
import random
import re
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import tiktoken


STOP_WORDS = {
    "def", "class", "import", "from", "return", "if", "else", "for",
    "while", "try", "except", "with", "as", "in", "not", "and", "or",
    "is", "None", "True", "False", "self", "cls", "lambda", "yield",
    "raise", "pass", "break", "continue", "global", "nonlocal", "del",
    "assert", "finally", "elif", "async", "await", "print", "len",
    "range", "str", "int", "float", "list", "dict", "set", "tuple",
    "type", "isinstance", "hasattr", "getattr", "setattr", "super",
    "property", "staticmethod", "classmethod", "abstractmethod", "Optional",
    "List", "Dict", "Set", "Tuple", "Any", "Union", "the", "that",
    "this", "are", "was", "were", "been", "has", "have", "had", "do",
    "does", "did", "will", "would", "can", "could", "should", "may",
    "might", "must", "need",
}


class TokenCounter:
    def __init__(self) -> None:
        self.encoder = tiktoken.get_encoding("cl100k_base")

    def count(self, text: str) -> int:
        return len(self.encoder.encode(text))


def extract_keywords(code: str) -> list[str]:
    return [
        token
        for token in re.findall(r"\b[a-zA-Z_]\w{2,}\b", code)
        if token not in STOP_WORDS
    ]


def extract_imports(code: str) -> set[str]:
    imports: set[str] = set()
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".")[0])
    except SyntaxError:
        imports.update(re.findall(r"^\s*import\s+(\w+)", code, re.MULTILINE))
        imports.update(re.findall(r"^\s*from\s+(\w+)", code, re.MULTILINE))
    return imports


class BM25Index:
    def __init__(self, corpus: list[dict], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.term_counts = [Counter(extract_keywords(d.get("text", ""))) for d in corpus]
        self.lengths = [sum(counts.values()) for counts in self.term_counts]
        self.avg_length = sum(self.lengths) / max(1, len(self.lengths))
        document_frequency: Counter[str] = Counter()
        for counts in self.term_counts:
            document_frequency.update(counts.keys())
        n = len(corpus)
        self.idf = {
            term: math.log((n - frequency + 0.5) / (frequency + 0.5) + 1)
            for term, frequency in document_frequency.items()
        }

    def rank(self, query: str) -> list[tuple[int, float]]:
        query_terms = extract_keywords(query)
        scored: list[tuple[int, float]] = []
        for index, counts in enumerate(self.term_counts):
            score = 0.0
            length = self.lengths[index]
            for term in query_terms:
                frequency = counts.get(term, 0)
                if not frequency:
                    continue
                denominator = frequency + self.k1 * (
                    1 - self.b + self.b * length / max(1, self.avg_length)
                )
                score += self.idf.get(term, 0.0) * frequency * (self.k1 + 1) / denominator
            if score > 0:
                scored.append((index, score))
        return sorted(scored, key=lambda item: (-item[1], item[0]))


class CoImportGraph:
    """Undirected shared-import/co-import similarity graph."""

    def __init__(self, corpus: list[dict]) -> None:
        module_docs: dict[str, set[int]] = defaultdict(set)
        for index, doc in enumerate(corpus):
            for module in extract_imports(doc.get("text", "")):
                module_docs[module].add(index)
        adjacency: dict[int, set[int]] = defaultdict(set)
        for indices in module_docs.values():
            for index in indices:
                adjacency[index].update(indices - {index})
        self.adjacency = adjacency
        self._cache: dict[tuple[int, int], tuple[tuple[int, int], ...]] = {}

    @property
    def edge_count(self) -> int:
        return sum(len(neighbors) for neighbors in self.adjacency.values()) // 2

    def related(self, start: int, max_hops: int) -> tuple[tuple[int, int], ...]:
        key = (start, max_hops)
        if key in self._cache:
            return self._cache[key]
        visited = {start}
        queue = deque([(start, 0)])
        result: list[tuple[int, int]] = []
        while queue:
            current, distance = queue.popleft()
            if distance == max_hops:
                continue
            for neighbor in sorted(self.adjacency.get(current, ())):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                result.append((neighbor, distance + 1))
                queue.append((neighbor, distance + 1))
        self._cache[key] = tuple(result)
        return self._cache[key]


class DirectedModuleGraph:
    """Directed file-level graph obtained by resolving local Python imports.

    The graph parses the checked-out repository rather than the overlapping
    retrieval chunks. An edge ``A -> B`` means that file A contains an import
    resolved to local file B. Unresolved third-party imports are not edges.
    """

    def __init__(self, corpus: list[dict], repo_root: Path, repo_key: str) -> None:
        self.repo_root = repo_root.resolve()
        self.repo_key = repo_key
        self.title_to_indices: dict[str, list[int]] = defaultdict(list)
        for index, doc in enumerate(corpus):
            self.title_to_indices[doc.get("title", "")].append(index)

        self.path_to_title: dict[Path, str] = {}
        self.module_to_path: dict[str, Path] = {}
        for path in sorted(self.repo_root.rglob("*.py")):
            if any(part in {".git", ".venv", "venv", "build", "dist"} for part in path.parts):
                continue
            relative = path.relative_to(self.repo_root)
            title = "-".join((repo_key, *relative.parts))
            if title not in self.title_to_indices:
                continue
            self.path_to_title[path] = title
            parts = list(relative.with_suffix("").parts)
            keys = [".".join(parts)]
            if parts and parts[-1] == "__init__":
                keys.append(".".join(parts[:-1]))
            if parts and parts[0] == "src":
                keys.append(".".join(parts[1:]))
                if parts[-1] == "__init__":
                    keys.append(".".join(parts[1:-1]))
            for key in keys:
                if key:
                    self.module_to_path.setdefault(key, path)

        self.adjacency: dict[str, set[str]] = defaultdict(set)
        self.import_statements = 0
        self.resolved_import_statements = 0
        for path, source_title in self.path_to_title.items():
            try:
                tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
            except SyntaxError:
                continue
            source_module = self._module_name(path)
            for node in ast.walk(tree):
                targets: list[str] = []
                if isinstance(node, ast.Import):
                    self.import_statements += 1
                    targets.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    self.import_statements += 1
                    base = self._absolute_from_module(source_module, node.module, node.level)
                    if base:
                        targets.extend(f"{base}.{alias.name}" for alias in node.names if alias.name != "*")
                        targets.append(base)
                else:
                    continue
                resolved = {self._resolve_module(target) for target in targets}
                resolved.discard(None)
                if resolved:
                    self.resolved_import_statements += 1
                for target_path in resolved:
                    target_title = self.path_to_title.get(target_path)
                    if target_title and target_title != source_title:
                        self.adjacency[source_title].add(target_title)

    def _module_name(self, path: Path) -> str:
        parts = list(path.relative_to(self.repo_root).with_suffix("").parts)
        if parts and parts[0] == "src":
            parts = parts[1:]
        if parts and parts[-1] == "__init__":
            parts = parts[:-1]
        return ".".join(parts)

    @staticmethod
    def _absolute_from_module(source: str, module: str | None, level: int) -> str:
        if level == 0:
            return module or ""
        source_parts = source.split(".") if source else []
        package = source_parts[:-1]
        keep = max(0, len(package) - (level - 1))
        base = package[:keep]
        if module:
            base.extend(module.split("."))
        return ".".join(base)

    def _resolve_module(self, module: str) -> Path | None:
        candidate = module
        while candidate:
            if candidate in self.module_to_path:
                return self.module_to_path[candidate]
            candidate = candidate.rpartition(".")[0]
        return None

    @property
    def edge_count(self) -> int:
        return sum(len(targets) for targets in self.adjacency.values())

    def outgoing_indices(self, index: int, corpus: list[dict]) -> list[int]:
        title = corpus[index].get("title", "")
        return [
            target_index
            for target_title in sorted(self.adjacency.get(title, ()))
            for target_index in self.title_to_indices.get(target_title, ())
        ]


@dataclass(frozen=True)
class RetrievalConfig:
    graph_boost: float = 0.3
    expansion_hops: int = 1
    enable_rerank: bool = True
    enable_expansion: bool = True
    initial_k_factor: int = 3


class Retriever:
    def __init__(self, corpus: list[dict], *, k1: float = 1.5, b: float = 0.75,
                 repo_root: Path | None = None, repo_key: str | None = None) -> None:
        self.corpus = corpus
        counter = TokenCounter()
        self.tokens = [counter.count(doc.get("text", "")) for doc in corpus]
        self.bm25 = BM25Index(corpus, k1=k1, b=b)
        self.graph = CoImportGraph(corpus)
        self.directed_graph = (
            DirectedModuleGraph(corpus, repo_root, repo_key)
            if repo_root is not None and repo_key is not None else None
        )
        self.average_tokens = sum(self.tokens) / max(1, len(self.tokens))

    def bm25_ranking(self, query: str) -> list[int]:
        return [index for index, _ in self.bm25.rank(query)]

    def gcc_ranking(self, query: str, budget: int, config: RetrievalConfig) -> list[int]:
        scored = self.bm25.rank(query)
        target = max(5, int(budget / max(1, self.average_tokens) * config.initial_k_factor))
        candidates = scored[:target]
        if not candidates:
            return []

        if config.enable_rerank and config.graph_boost > 0 and config.expansion_hops > 0:
            candidate_ids = {index for index, _ in candidates}
            boosts: dict[int, float] = defaultdict(float)
            for seed, _ in candidates[:20]:
                for related, distance in self.graph.related(seed, config.expansion_hops):
                    if related in candidate_ids:
                        boosts[related] += config.graph_boost / (1 + distance)
            candidates = sorted(
                ((index, score + boosts.get(index, 0.0)) for index, score in candidates),
                key=lambda item: (-item[1], item[0]),
            )

        if config.enable_expansion and config.expansion_hops > 0:
            expanded = list(candidates)
            for seed, score in candidates[:10]:
                for related, distance in self.graph.related(seed, config.expansion_hops):
                    expanded.append((related, score * 0.3 / (1 + distance)))
            best: dict[int, float] = {}
            for index, score in expanded:
                best[index] = max(best.get(index, float("-inf")), score)
            candidates = sorted(best.items(), key=lambda item: (-item[1], item[0]))
        return [index for index, _ in candidates]

    def directed_ranking(self, query: str, budget: int, graph_boost: float = 0.3) -> list[int]:
        """BM25 followed by expansion along resolved outgoing module imports."""
        if self.directed_graph is None:
            raise RuntimeError("directed module graph was not configured")
        scored = self.bm25.rank(query)
        score_by_index = dict(scored)
        seeds = [index for index, _ in scored[:20]]
        boosted = dict(scored)
        for seed_rank, seed in enumerate(seeds):
            seed_score = score_by_index[seed]
            for target in self.directed_graph.outgoing_indices(seed, self.corpus):
                # Prefer lexically relevant chunks within a dependent file but
                # retain a small structural score when lexical overlap is zero.
                structural = graph_boost * seed_score / (seed_rank + 1)
                boosted[target] = max(boosted.get(target, 0.0), score_by_index.get(target, 0.0) + structural)
        return [index for index, _ in sorted(boosted.items(), key=lambda item: (-item[1], item[0]))]

    def random_ranking(self, query_id: str, seed: int) -> list[int]:
        digest = hashlib.sha256(f"{query_id}:{seed}".encode()).digest()
        rng = random.Random(int.from_bytes(digest[:8], "big"))
        ranking = list(range(len(self.corpus)))
        rng.shuffle(ranking)
        return ranking

    def select(self, ranking: Iterable[int], budget: int) -> list[int]:
        """Pack whole chunks without ever exceeding ``budget``.

        Oversized chunks are skipped so a later smaller chunk may still fit.
        """
        selected: list[int] = []
        used = 0
        for index in ranking:
            size = self.tokens[index]
            if used + size <= budget:
                selected.append(index)
                used += size
        assert used <= budget
        return selected


def recall(selected: list[str], relevant: set[str]) -> float:
    return len(set(selected) & relevant) / len(relevant) if relevant else 0.0


def precision(selected: list[str], relevant: set[str]) -> float:
    return len(set(selected) & relevant) / len(set(selected)) if selected else 0.0


def f1(selected: list[str], relevant: set[str]) -> float:
    p, r = precision(selected, relevant), recall(selected, relevant)
    return 2 * p * r / (p + r) if p + r else 0.0


def ndcg(selected: list[str], relevant: set[str], k: int = 10) -> float:
    dcg = sum(
        1 / math.log2(rank + 2)
        for rank, doc_id in enumerate(selected[:k])
        if doc_id in relevant
    )
    ideal = min(k, len(relevant))
    idcg = sum(1 / math.log2(rank + 2) for rank in range(ideal))
    return dcg / idcg if idcg else 0.0


def mrr(selected: list[str], relevant: set[str]) -> float:
    for rank, doc_id in enumerate(selected):
        if doc_id in relevant:
            return 1 / (rank + 1)
    return 0.0
