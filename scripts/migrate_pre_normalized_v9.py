#!/usr/bin/env python3
"""Migrate only prompt-identical rows from the exact-only V9 pilot.

The old pilot used exact-string masking.  The current blind inputs use
whitespace-insensitive target masking, so successful generations are reusable
only when their effective user prompt is byte-identical.  Old source files
remain archived under ``*_pre_normalized``; this script writes only the
verified overlap into the active V9 output directories.
"""

from __future__ import annotations

import importlib.util
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLD_INPUTS = ROOT / "results/generation_inputs/blinded_pre_ranking_exact_only"
NEW_INPUTS = ROOT / "results/generation_inputs/blinded_pre_ranking"
OLD_RAW = ROOT / "results/downstream/v9_pre_normalized"
NEW_RAW = ROOT / "results/downstream/v9"
OLD_EXEC = ROOT / "results/execution/v9_pre_normalized"
NEW_EXEC = ROOT / "results/execution/v9"
STRATEGIES = ("bm25", "coimport", "directed")
MODELS = {
    "deepseek": "deepseek-v4.1-flash",
    "muse": "muse-spark-1.3-contributor",
}


def load_runner():
    spec = importlib.util.spec_from_file_location(
        "downstream_runner", ROOT / "scripts/run_downstream_generation.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n"
                                   for row in rows))


def input_hashes(runner, directory: Path) -> dict[tuple[str, str], str]:
    hashes = {}
    for strategy in STRATEGIES:
        rows = load_jsonl(directory / f"repoeval_function_{strategy}_4k.jsonl")
        if len(rows) != 341:
            raise SystemExit(f"Expected 341 {strategy} inputs in {directory}")
        for row in rows:
            key = (row["metadata"]["task_id"], strategy)
            digest = runner.prompt_sha256(row)
            if key in hashes:
                raise SystemExit(f"Duplicate input key: {key}")
            hashes[key] = digest
    return hashes


def main() -> None:
    for directory in (NEW_RAW, NEW_EXEC):
        if directory.exists() and any(directory.iterdir()):
            raise SystemExit(f"Refusing to overwrite non-empty directory: {directory}")
    runner = load_runner()
    old_hashes = input_hashes(runner, OLD_INPUTS)
    new_hashes = input_hashes(runner, NEW_INPUTS)
    if old_hashes.keys() != new_hashes.keys():
        raise SystemExit("Old and new generation input keys differ")

    changed = {key for key in old_hashes if old_hashes[key] != new_hashes[key]}
    reusable: dict[str, set[tuple[str, str, str]]] = {}
    audit = {
        "source_variant": "blinded_pre_ranking_exact_only",
        "active_variant": "blinded_pre_ranking",
        "reuse_rule": "successful rows only when effective user-prompt SHA-256 matches",
        "old_input_rows": len(old_hashes),
        "new_input_rows": len(new_hashes),
        "changed_prompt_key_count": len(changed),
        "changed_prompt_keys": [
            {"task_id": task, "strategy": strategy,
             "old_sha256": old_hashes[(task, strategy)],
             "new_sha256": new_hashes[(task, strategy)]}
            for task, strategy in sorted(changed)],
        "models": {},
        "execution": {},
    }

    for short, model in MODELS.items():
        source = OLD_RAW / f"raw_{short}.jsonl"
        rows = load_jsonl(source)
        successful = [r for r in rows
                      if r.get("error") is None and r.get("completion") is not None]
        errors = [r for r in rows if r.get("error") is not None]
        migrated = []
        invalidated = []
        seen = set()
        for row in successful:
            if row["model"] != model:
                raise SystemExit(f"Unexpected model in {source}: {row['model']}")
            key = (row["task_id"], row["strategy"], model)
            if key in seen:
                raise SystemExit(f"Duplicate successful source key: {key}")
            seen.add(key)
            prompt_key = (row["task_id"], row["strategy"])
            if prompt_key in changed:
                invalidated.append(key)
                continue
            copy = dict(row)
            copy["input_sha256"] = new_hashes[prompt_key]
            migrated.append(copy)
        if any(r["model"] != model for r in errors):
            raise SystemExit(f"Unexpected error-row model in {source}")
        output = NEW_RAW / f"raw_{short}.jsonl"
        write_jsonl(output, migrated)
        reusable[short] = {(r["task_id"], r["strategy"], model)
                           for r in migrated}
        per_strategy = Counter(r["strategy"] for r in migrated)
        audit["models"][short] = {
            "model_id": model,
            "source_rows": len(rows),
            "source_successes": len(successful),
            "source_errors_excluded": len(errors),
            "prompt_changed_successes_excluded": len(invalidated),
            "reused_successes": len(migrated),
            "missing_current_matrix_rows": 1023 - len(migrated),
            "reused_by_strategy": {s: per_strategy[s] for s in STRATEGIES},
            "invalidated_keys": [
                {"task_id": task, "strategy": strategy}
                for task, strategy, _ in sorted(invalidated)],
        }

    for short, model in MODELS.items():
        allowed = reusable[short]
        model_counts = Counter()
        for source in sorted(OLD_EXEC.glob(f"eval_{short}_w*.jsonl")):
            rows = load_jsonl(source)
            kept = [r for r in rows
                    if (r["task_id"], r["strategy"], r["model"]) in allowed]
            if any(r["model"] != model for r in rows):
                raise SystemExit(f"Unexpected model in execution shard {source}")
            name = source.name
            write_jsonl(NEW_EXEC / name, kept)
            model_counts.update(r["strategy"] for r in kept)
        expected = len(allowed)
        actual = sum(model_counts.values())
        if actual != expected:
            raise SystemExit(
                f"Execution reuse mismatch for {short}: {actual} != {expected}")
        audit["execution"][short] = {
            "reused_rows": actual,
            "reused_by_strategy": {s: model_counts[s] for s in STRATEGIES},
            "matches_reused_generation_keys": True,
        }

    audit_path = NEW_RAW / "reuse_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "changed_prompt_keys": len(changed),
        "reused_successes": {
            name: audit["models"][name]["reused_successes"] for name in MODELS},
        "remaining": {
            name: audit["models"][name]["missing_current_matrix_rows"]
            for name in MODELS},
    }, indent=2))


if __name__ == "__main__":
    main()
