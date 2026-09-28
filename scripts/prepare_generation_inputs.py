#!/usr/bin/env python3
"""Create CodeRAG-Bench-compatible RepoEval generation inputs at 4K tokens."""

from __future__ import annotations

import json
import re
from pathlib import Path

from gcc import RetrievalConfig, Retriever, TokenCounter
from run_downstream_generation import prompt_sha256


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "scripts" / "code-rag-bench" / "retrieval" / "datasets"
SOURCES = ROOT / "scripts" / "code-rag-bench" / "retrieval" / "output" / "repoeval" / "repositories" / "function_level"
OUTPUT = ROOT / "results" / "generation_inputs" / "blinded_pre_ranking"
BUDGET = 4000


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize_whitespace(text: str) -> str:
    """Remove all whitespace, so line wrapping and indentation do not matter."""
    return re.sub(r"\s+", "", text)


def normalized_match_spans(text: str, reference: str) -> list[tuple[int, int]]:
    """Locate reference occurrences even when whitespace differs."""
    target = normalize_whitespace(reference.strip())
    if not target:
        return []

    positions: list[int] = []
    characters: list[str] = []
    for index, character in enumerate(text):
        if not character.isspace():
            positions.append(index)
            characters.append(character)
    normalized = "".join(characters)

    spans: list[tuple[int, int]] = []
    cursor = 0
    while True:
        start = normalized.find(target, cursor)
        if start < 0:
            break
        end = start + len(target)
        spans.append((positions[start], positions[end - 1] + 1))
        cursor = start + 1

    if not spans:
        return []

    # Merge overlapping normalized matches before replacing source spans.
    merged = [spans[0]]
    for start, end in spans[1:]:
        previous_start, previous_end = merged[-1]
        if start <= previous_end:
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return merged


def mask_target_span(text: str, reference: str) -> str:
    """Mask every full-reference occurrence regardless of whitespace layout."""
    spans = normalized_match_spans(text, reference)
    for start, end in reversed(spans):
        text = text[:start] + "# <redacted-target-span>" + text[end:]
    return text


def audit_no_leakage(outputs: dict[str, list[dict]]) -> dict:
    """Scan built inputs for recoverable full references (R-18-V8).

    Two levels: (1) exact stripped-reference occurrence in selected docs or
    in the decommented model prompt; (2) whitespace-normalized occurrence.
    Both levels must be clear in the pre-ranking blinded inputs.
    """
    import sys as _sys

    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_downstream_generation import get_retrieved_prompt
    report: dict[str, object] = {"exact_hits": [], "normalized_hits": []}
    for strategy, rows in outputs.items():
        for row in rows:
            reference = row.get("reference", "")
            stripped = reference.strip()
            if not stripped:
                continue
            docs = row.get("docs", [])
            if any(stripped in doc.get("text", "") for doc in docs):
                report["exact_hits"].append(
                    (row["metadata"]["task_id"], strategy))  # type: ignore[union-attr]
                continue
            prompt = get_retrieved_prompt(docs)

            def decomment(line: str) -> str:
                return line[2:] if line.startswith("# ") else line

            clean = "\n".join(decomment(x) for x in prompt.split("\n"))
            nref = normalize_whitespace(stripped)
            if nref and (nref in normalize_whitespace(clean)
                         or nref in normalize_whitespace(
                             "\n".join(doc.get("text", "") for doc in docs))):
                report["normalized_hits"].append(  # type: ignore[union-attr]
                    (row["metadata"]["task_id"], strategy))
    exact = report["exact_hits"]
    normalized = report["normalized_hits"]
    assert isinstance(exact, list)
    if exact:
        raise SystemExit(f"LEAKAGE: {len(exact)} exact full-reference hits, "
                         f"e.g. {exact[:3]}")
    assert isinstance(normalized, list)
    if normalized:
        raise SystemExit(
            f"LEAKAGE: {len(normalized)} whitespace-normalized full-reference "
            f"hits remain in blinded prompts, e.g. {normalized[:3]}")
    return report


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    outputs = {strategy: [] for strategy in ("bm25", "coimport", "directed")}
    candidate_masking_audit = {
        "normalization": "remove all whitespace before matching the full reference",
        "tasks_with_candidate_matches": [],
        "candidate_chunks_with_matches_before_masking": 0,
        "remaining_full_reference_hits": [],
    }
    for repo_dir in sorted(DATA.glob("repoeval__*")):
        repo_key = repo_dir.name.removeprefix("repoeval__")
        corpus = read_jsonl(repo_dir / "corpus.jsonl")
        queries = read_jsonl(repo_dir / "queries.jsonl")
        for query in queries:
            query_text = query.get("text", "")
            reference = query.get("metadata", {}).get("ground_truth", "")
            # R-18-V8: mask the target span in the CANDIDATE corpus BEFORE
            # building the retriever, so rankings and budget packing are
            # computed on the same blinded text the model will see.
            # Masked chunks keep their corpus position (BM25 length norm and
            # graph adjacency stay comparable); only their text changes.
            target = normalize_whitespace(reference.strip())
            matching_ids = [
                doc.get("_id", "") for doc in corpus
                if target and target in normalize_whitespace(doc.get("text", ""))
            ]
            candidate_masking_audit[
                "candidate_chunks_with_matches_before_masking"] += len(matching_ids)
            if matching_ids:
                candidate_masking_audit["tasks_with_candidate_matches"].append({
                    "task_id": query.get("metadata", {}).get("task_id"),
                    "candidate_chunk_ids": matching_ids,
                })
            masked_corpus = []
            for doc in corpus:
                masked_text = mask_target_span(doc.get("text", ""), reference)
                if target and target in normalize_whitespace(masked_text):
                    candidate_masking_audit["remaining_full_reference_hits"].append({
                        "task_id": query.get("metadata", {}).get("task_id"),
                        "candidate_chunk_id": doc.get("_id", ""),
                    })
                masked_corpus.append(dict(doc, text=masked_text))
            n_masked_corpus = sum(
                1 for orig, new in zip(corpus, masked_corpus)
                if orig.get("text", "") != new["text"])
            retriever = Retriever(masked_corpus, repo_root=SOURCES / repo_key,
                                  repo_key=repo_key)
            # Token accounting uses the masked text the model actually sees.
            masked_tokens = TokenCounter()
            masked_sizes = [masked_tokens.count(doc.get("text", ""))
                            for doc in masked_corpus]
            rankings = {
                "bm25": retriever.bm25_ranking(query_text),
                "coimport": retriever.gcc_ranking(query_text, BUDGET, RetrievalConfig()),
                "directed": retriever.directed_ranking(query_text, BUDGET),
            }
            for strategy, ranking in rankings.items():
                # Pack against masked sizes so the budget binds model-visible
                # tokens (R-18-V8 audit: mean -85/strategy/task vs unmasked).
                selected: list[int] = []
                used = 0
                for index in ranking:
                    size = masked_sizes[index]
                    if used + size <= BUDGET:
                        selected.append(index)
                        used += size
                assert used <= BUDGET
                docs = [
                    {"id": masked_corpus[index]["_id"],
                     "title": masked_corpus[index].get("title", ""),
                     "text": masked_corpus[index].get("text", "")}
                    for index in selected
                ]
                input_row = {
                    "prompt": query_text,
                    "reference": reference,
                    "metadata": query.get("metadata", {}),
                    "docs": docs,
                    "retrieval_strategy": strategy,
                    "retrieved_tokens_cl100k": used,
                    "masked_chunks": n_masked_corpus,
                }
                input_row["input_sha256"] = prompt_sha256(input_row)
                outputs[strategy].append(input_row)
    if candidate_masking_audit["remaining_full_reference_hits"]:
        raise SystemExit(
            "LEAKAGE: normalized target references remain in candidate corpus: "
            f"{candidate_masking_audit['remaining_full_reference_hits'][:3]}")
    with (OUTPUT / "candidate_masking_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(candidate_masking_audit, handle, indent=2, sort_keys=True)
    for strategy, rows in outputs.items():
        path = OUTPUT / f"repoeval_function_{strategy}_4k.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        print(path, len(rows))
    report = audit_no_leakage(outputs)
    print("leakage audit: exact=0",
          f"normalized={len(report['normalized_hits'])}",
          report["normalized_hits"][:5])
    with (OUTPUT / "leakage_audit.json").open("w", encoding="utf-8") as handle:
        json.dump({"normalized_hits": sorted(report["normalized_hits"]),
                   "n_exact": 0,
                   "n_normalized": len(report["normalized_hits"])},
                  handle, indent=2)


if __name__ == "__main__":
    main()
