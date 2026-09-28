#!/usr/bin/env python3
"""Downstream generation runner: feed prepared RepoEval prompts to an
instruction-tuned code model via the OpenCode Zen chat-completions API.

Fairness: the message wrapper is byte-identical for every strategy; only the
stored prompt (retrieval context) differs. Greedy-equivalent decoding
(temperature 0, fixed seed). Raw completions + usage saved per row.

Key: read from `.env` (OPENCODE_APIKEY); never printed or logged.

Usage:
  python3 scripts/run_downstream_generation.py --model MODEL \
      --inputs results/generation_inputs/reported_post_selection \
      --out results/downstream/v8/raw_MODEL.jsonl
  python3 scripts/run_downstream_generation.py --model MODEL --limit 6 ...  # pilot
"""

import argparse
import hashlib
import json
import os
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = "https://opencode.ai/zen/go/v1"
SESSION = "dsh-ai-se-lab-downstream"
STRATEGIES = ["bm25", "coimport", "directed"]
# Input variant used by the reported experiment: the selector ranks and packs
# on the unmodified corpus, then the selected documents are redacted. The
# blinded ordering lives in results/generation_inputs/blinded_pre_ranking/
# (see scripts/audit_masking_order.py and paper section 3.5).
DEFAULT_INPUTS = "results/generation_inputs/reported_post_selection"
FILES = {s: f"repoeval_function_{s}_4k.jsonl" for s in STRATEGIES}

SYSTEM = ("You are a Python code completion engine. The user message is a "
          "Python file prefix ending in the middle of a function body. "
          "Output ONLY the continuation code (the missing function body "
          "lines), with no explanations, no markdown fences, no extra text.")
WRAPPER_SUFFIX = ("\n\n# TASK: continue the code above. "
                  "Output only the missing continuation lines.")


def get_retrieved_prompt(docs):
    """Verbatim port of CodeRAG-Bench's get_retrieved_prompt
    (scripts/code-rag-bench/generation/eval/tasks/repoeval.py), so the
    model sees exactly the official prompt construction."""
    start_line = "Here are some relevant code fragments from other files of the repo:"
    sep_line = "--------------------------------------------------"
    intro_line = "the below code fragment can be found in:"

    title_block = intro_line + '\n' + '__TITLE__' + '\n' + sep_line

    retrieved_prompt = start_line + '\n' + sep_line + '\n'
    for doc in docs:
        title, text = doc['title'], doc['text']
        retrieved_prompt += title_block.replace('__TITLE__', title) + '\n'
        retrieved_prompt += doc['text'] + '\n' + sep_line + '\n'

    # add '# ' to each line except for the last line
    retrieved_prompt = "\n".join(
        ["# " + x for x in retrieved_prompt.split('\n')[:-1]]
    ) + '\n'
    return retrieved_prompt


def build_prompt(row):
    docs = row.get("docs", [])
    if docs:
        return get_retrieved_prompt(docs) + '\n\n' + row["prompt"]
    return row["prompt"]


def prompt_sha256(row):
    """Fingerprint the user prompt so resume cannot reuse a changed input."""
    return hashlib.sha256(build_prompt(row).encode("utf-8")).hexdigest()


def load_key():
    for line in open(".env"):
        line = line.strip()
        if line.startswith("OPENCODE_APIKEY="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError("OPENCODE_APIKEY not found in .env")


def complete(key, prompt, model, max_tokens, timeout, api="chat"):
    if api == "responses":
        return complete_responses(key, prompt, model, max_tokens, timeout)
    body = json.dumps({
        "model": model,
        "temperature": 0,
        "seed": 20260913,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt + WRAPPER_SUFFIX},
        ],
    }).encode()
    req = urllib.request.Request(
        BASE + "/chat/completions", data=body,
        headers={"Authorization": "Bearer " + key,
                 "Content-Type": "application/json",
                 # NOTE: the gateway rejects Python-urllib's default UA with
                 # 403; use a neutral UA (verified 2026-09-13).
                 "User-Agent": "curl/8.0",
                 "x-opencode-session": SESSION})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.load(r)
        msg = d["choices"][0]["message"]["content"]
        return {"completion": msg, "usage": d.get("usage", {}),
                "latency_s": round(time.time() - t0, 2), "error": None}
    except Exception as e:  # noqa: BLE001 - recorded per row, retried by caller
        return {"completion": None, "usage": {},
                "latency_s": round(time.time() - t0, 2),
                "error": f"{type(e).__name__}: {e}"[:300]}


def complete_responses(key, prompt, model, max_tokens, timeout):
    """OpenAI Responses API variant (required for muse-spark-1.3-contributor,
    whose chat-completions route returns HTTP 500). Wrapper text is identical
    to the chat variant: SYSTEM goes in `instructions`, prompt+WRAPPER_SUFFIX
    in `input`. reasoning.effort=minimal (verified cheapest supported)."""
    body = json.dumps({
        "model": model,
        "temperature": 0,
        "max_output_tokens": max_tokens,
        "reasoning": {"effort": "minimal"},
        "instructions": SYSTEM,
        "input": prompt + WRAPPER_SUFFIX,
    }).encode()
    req = urllib.request.Request(
        BASE + "/responses", data=body,
        headers={"Authorization": "Bearer " + key,
                 "Content-Type": "application/json",
                 "User-Agent": "curl/8.0",
                 "x-opencode-session": SESSION})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.load(r)
        if d.get("status") != "completed" or d.get("error"):
            raise RuntimeError(f"bad status: {d.get('status')} {d.get('error')}")
        text = ""
        for o in d.get("output", []):
            if o.get("type") == "message":
                for c in o.get("content", []):
                    if c.get("type") == "output_text":
                        text += c.get("text", "")
        u = d.get("usage", {})
        return {"completion": text,
                "usage": {"prompt_tokens": u.get("input_tokens"),
                          "completion_tokens": u.get("output_tokens"),
                          "total_tokens": u.get("total_tokens")},
                "latency_s": round(time.time() - t0, 2), "error": None}
    except Exception as e:  # noqa: BLE001
        return {"completion": None, "usage": {},
                "latency_s": round(time.time() - t0, 2),
                "error": f"{type(e).__name__}: {e}"[:300]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--inputs", default=DEFAULT_INPUTS)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel API workers (rows are keyed, order-free)")
    ap.add_argument("--api", choices=["chat", "responses"], default="chat",
                    help="chat/completions (default) or responses API")
    a = ap.parse_args()

    api_key = load_key()
    tasks = []
    for s in STRATEGIES:
        with open(os.path.join(a.inputs, os.path.basename(FILES[s]))) as f:
            rows = [json.loads(l) for l in f]
        for row in rows:
            recorded_hash = row.get("input_sha256")
            if recorded_hash and recorded_hash != prompt_sha256(row):
                raise SystemExit(
                    f"Input manifest mismatch for "
                    f"{row['metadata']['task_id']}/{s}")
        for r in rows:
            tasks.append((s, r))
    tasks = tasks[a.offset:(a.offset + a.limit if a.limit else None)]
    print(f"model={a.model} tasks={len(tasks)}", flush=True)

    done_keys = {}
    if os.path.exists(a.out):
        with open(a.out) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    if d.get("error") is None:
                        done_key = (d["task_id"], d["strategy"], d["model"])
                        previous = done_keys.get(done_key)
                        current_hash = d.get("input_sha256")
                        if (previous and previous.get("input_sha256")
                                and current_hash
                                and previous["input_sha256"] != current_hash):
                            raise SystemExit(
                                f"Conflicting successful inputs for key {done_key}")
                        done_keys[done_key] = d
                except ValueError:
                    continue
    mode = "a" if os.path.exists(a.out) else "w"
    todo = []
    for strategy, row in tasks:
        row_key = (row["metadata"]["task_id"], strategy, a.model)
        previous = done_keys.get(row_key)
        if previous is None:
            todo.append((strategy, row))
            continue
        expected_hash = prompt_sha256(row)
        previous_hash = previous.get("input_sha256")
        if previous_hash is not None and previous_hash != expected_hash:
            raise SystemExit(
                f"Input changed for completed key {row_key}; archive or remove "
                "the stale row before resuming")
    print(f"resuming: {len(done_keys)} done, {len(todo)} remaining", flush=True)
    lock = threading.Lock()
    quota_limited = threading.Event()
    counter = [0]

    def one(item):
        if quota_limited.is_set():
            return
        s, r = item
        prompt = build_prompt(r)
        input_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        res = None
        for attempt in range(a.retries):
            if quota_limited.is_set():
                return
            res = complete(api_key, prompt, a.model,
                           a.max_tokens, a.timeout, api=a.api)
            if res["error"] is None:
                break
            # A provider quota error is shared by all workers. Stop this run
            # and resume from the saved successful keys after quota resets.
            if "429" in (res["error"] or ""):
                quota_limited.set()
                break
            if attempt + 1 < a.retries:
                time.sleep(5 * (attempt + 1))
        row = json.dumps({
            "task_id": r["metadata"]["task_id"],
            "strategy": s,
            "model": a.model,
            "api": a.api,
            "reference": r["reference"],
            "metadata": r["metadata"],
            "retrieved_tokens_cl100k": r["retrieved_tokens_cl100k"],
            "input_sha256": input_hash,
            **res,
        }) + "\n"
        with lock:
            out.write(row)
            out.flush()
            counter[0] += 1
            if counter[0] % 20 == 0:
                print(f"  {counter[0]}/{len(todo)}", flush=True)

    with open(a.out, mode) as out:
        if a.workers > 1:
            with ThreadPoolExecutor(max_workers=a.workers) as ex:
                list(ex.map(one, todo))
        else:
            for item in todo:
                one(item)
    print("quota-limited" if quota_limited.is_set() else "finished",
          a.out, flush=True)
    if quota_limited.is_set():
        raise SystemExit(75)


if __name__ == "__main__":
    main()
