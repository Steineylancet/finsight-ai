"""
FinSight AI — offline evaluation against the golden set.

Runs every case through the real LangGraph agent (live Azure OpenAI + AI Search)
and scores:
  routing       intent matches the expected path
  parameters    department / fiscal year / quarter / threshold extracted correctly
  behaviour     expected tools called, key facts present, clarifications raised
  grounding     every $ and % figure in the answer traces to tool/document output
  faithfulness  LLM judge (gpt-5.4-mini, a different model from the GPT-4o answer
                writer, to avoid self-grading) scores support by the evidence, 1–5

Usage:  python -m evals.run_evals [--only id1,id2] [--no-judge]
Costs a few tens of cents in tokens per full run. Exit code 1 if any gate fails.
"""

import argparse
import itertools
import json
import re
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from pydantic import BaseModel, Field  # noqa: E402

from backend.data_loader import DataLoader  # noqa: E402
from backend.graph.catalog import resolve_department  # noqa: E402
from backend.graph.llm import mini_llm  # noqa: E402
from backend.graph.router import build_agent_graph  # noqa: E402
from backend.rag_pipeline import RAGPipeline  # noqa: E402

ROOT = Path(__file__).parent
GATES = {"routing": 0.90, "parameters": 0.90, "behaviour": 0.90, "grounding": 0.95, "faithfulness": 4.0}
PAUSE_BETWEEN_CASES_S = 2  # GPT-4o deployment is 10K TPM — pace to stay under it


# ─────────────────────────────────────────────────────────────────────────────
# Evidence collection
# ─────────────────────────────────────────────────────────────────────────────

def _walk(obj):
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _walk(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _walk(v)
    else:
        yield obj


def evidence_text(result: dict) -> str:
    return json.dumps(result.get("tool_results") or {}, default=str) + "\n" + (result.get("formatted_table") or "")


def tool_calls(result: dict) -> list[dict]:
    tr = result.get("tool_results") or {}
    if "calls" in tr:
        return tr["calls"]
    if result.get("sql_query"):
        name = result["sql_query"].split("(", 1)[0]
        args = json.loads(result["sql_query"][len(name) + 1:-1])
        return [{"name": name, "args": args}]
    return []


# ─────────────────────────────────────────────────────────────────────────────
# Number grounding
# ─────────────────────────────────────────────────────────────────────────────

_SCALE = {"k": 1e3, "thousand": 1e3, "m": 1e6, "mm": 1e6, "million": 1e6, "b": 1e9, "bn": 1e9, "billion": 1e9}
_MONEY = re.compile(r"\$\s?([+-]?[\d,]*\.?\d+)\s?(million|billion|thousand|mm|bn|[kmb])?\b", re.I)
_PCT = re.compile(r"([+-]?\d+(?:\.\d+)?)\s?%")
_ANY_NUM = re.compile(r"[+-]?\d[\d,]*\.?\d*")


def _num(s: str) -> float:
    return float(s.replace(",", ""))


def _decimals(s: str) -> int:
    return len(s.split(".")[1]) if "." in s else 0


def claimed_figures(text: str) -> list[tuple[str, float, float, str]]:
    """(raw, value, tolerance, kind) for each $ and % figure in the answer."""
    out = []
    for m in _MONEY.finditer(text):
        raw, scale = m.group(1), _SCALE.get((m.group(2) or "").lower(), 1)
        tol = 0.51 * 10 ** -_decimals(raw) * scale + 0.01
        out.append((m.group(0), abs(_num(raw)) * scale, tol, "money"))
    for m in _PCT.finditer(text):
        raw = m.group(1)
        out.append((m.group(0), abs(_num(raw)), 0.51 * 10 ** -_decimals(raw) + 0.001, "pct"))
    return out


def source_numbers(result: dict) -> set[float]:
    nums = set()
    for leaf in _walk(result.get("tool_results") or {}):
        if isinstance(leaf, bool):
            continue
        if isinstance(leaf, (int, float)) and leaf == leaf:
            nums.add(abs(float(leaf)))
        elif isinstance(leaf, str):
            nums.update(abs(_num(n)) for n in _ANY_NUM.findall(leaf) if n.strip("+-,."))
    for n in _ANY_NUM.findall(result.get("formatted_table") or ""):
        if n.strip("+-,."):
            nums.add(abs(_num(n)))
    return nums


def grounding(result: dict) -> tuple[int, int, list[str]]:
    text = result.get("narrative") or ""
    claims = claimed_figures(text)
    if not claims:
        return 0, 0, []
    src = sorted(source_numbers(result))
    big = [s for s in src if s >= 1000][:150]
    unsupported = []
    for raw, value, tol, kind in claims:
        if any(abs(s - value) <= tol for s in src):
            continue
        # allow simple arithmetic the writer may do: ratios of counts ("26 of 26" → 100%),
        # and sums/differences of two or three dollar figures
        counts = [s for s in src if s == int(s) and 0 < s <= 1000]
        if kind == "pct" and any(abs(a / b * 100 - value) <= tol for a in counts for b in counts if a <= b):
            continue
        if kind == "money" and value >= 1000 and (
            any(abs((a + b) - value) <= tol or abs(abs(a - b) - value) <= tol
                for a, b in itertools.combinations(big, 2))
            or any(abs((a + b + c) - value) <= tol for a, b, c in itertools.combinations(big[:60], 3))
        ):
            continue
        unsupported.append(raw)
    return len(claims) - len(unsupported), len(claims), unsupported


# ─────────────────────────────────────────────────────────────────────────────
# Faithfulness judge
# ─────────────────────────────────────────────────────────────────────────────

class Judgement(BaseModel):
    score: int = Field(ge=1, le=5, description="5 = every claim supported by the evidence; 1 = mostly unsupported.")
    unsupported_claims: list[str] = Field(description="Claims not supported by the evidence (empty if none).")


def judge(question: str, answer: str, evidence: str) -> Judgement:
    prompt = f"""You are auditing a financial assistant's answer for faithfulness.
Score 1–5 how well EVERY factual claim in the answer is supported by the evidence.
Generic framing, hedges ("further analysis needed") and restating the question are fine.
Correct arithmetic on evidence figures counts as supported. Penalise invented numbers,
invented causes presented as fact, and policy statements not in the evidence.

QUESTION: {question}

EVIDENCE:
{evidence[:14000]}

ANSWER:
{answer}"""
    return mini_llm().with_structured_output(Judgement).invoke(prompt)


# ─────────────────────────────────────────────────────────────────────────────
# Case scoring
# ─────────────────────────────────────────────────────────────────────────────

def _norm_dept(v):
    res = resolve_department(v) if v else None
    return res.department if res and res.ok else v


def score_case(case: dict, result: dict) -> dict:
    exp = case["expect"]
    checks = {}

    if "intent" in exp:
        checks["routing"] = result.get("intent") == exp["intent"]

    calls = tool_calls(result)
    for key in ("department", "fiscal_year", "quarter", "threshold_pct"):
        if key not in exp:
            continue
        seen = {result.get(key)} | {c.get("args", {}).get(key) for c in calls}
        if key == "department":
            ok = exp[key] in {_norm_dept(v) for v in seen if v}
        elif key == "threshold_pct":
            ok = any(v is not None and float(v) == float(exp[key]) for v in seen)
        else:
            ok = str(exp[key]) in {str(v) for v in seen if v}
        checks[f"param:{key}"] = ok

    names = {c["name"] for c in calls}
    answer = (result.get("narrative") or "") + " " + (result.get("error") or "")
    if "tools" in exp:
        checks["behaviour:tools"] = set(exp["tools"]) <= names
    if "tools_any" in exp:
        checks["behaviour:tools_any"] = bool(set(exp["tools_any"]) & names)
    if "contains_any" in exp:
        checks["behaviour:contains"] = any(s.lower() in answer.lower() for s in exp["contains_any"])
    if "error_contains" in exp:
        checks["behaviour:clarifies"] = any(s.lower() in (result.get("error") or "").lower()
                                            for s in exp["error_contains"])
    elif result.get("error"):
        checks["behaviour:no_error"] = False
    return checks


def run(only: set[str] | None, use_judge: bool) -> int:
    cases = json.loads((ROOT / "golden_set.json").read_text(encoding="utf-8"))
    if only:
        cases = [c for c in cases if c["id"] in only]

    DataLoader.get()
    graph = build_agent_graph(RAGPipeline())

    rows = []
    for i, case in enumerate(cases, 1):
        t0 = time.time()
        try:
            result = graph.invoke({"query": case["question"], "conversation_history": case.get("history", [])})
        except Exception as e:  # a crash is a failed case, not a failed run
            result = {"error": f"EXCEPTION: {type(e).__name__}: {e}"}
        latency = time.time() - t0

        checks = score_case(case, result)
        g_ok, g_total, unsupported = grounding(result)
        verdict = None
        if use_judge and result.get("narrative") and (result.get("tool_results") or {}):
            try:
                verdict = judge(case["question"], result["narrative"], evidence_text(result))
            except Exception as e:
                print(f"   judge failed: {e}")

        row = {
            "id": case["id"], "category": case["category"], "question": case["question"],
            "intent": result.get("intent"), "mode": result.get("mode_label"),
            "standalone_query": result.get("standalone_query"),
            "tools": [c["name"] for c in tool_calls(result)],
            "checks": checks, "grounding": [g_ok, g_total], "unsupported_figures": unsupported,
            "faithfulness": verdict.score if verdict else None,
            "unsupported_claims": verdict.unsupported_claims if verdict else [],
            "latency_s": round(latency, 1),
            "error": result.get("error"), "narrative": result.get("narrative"),
        }
        rows.append(row)
        failed = [k for k, v in checks.items() if not v]
        status = "PASS" if not failed and not unsupported else "FAIL"
        print(f"[{i:02}/{len(cases)}] {status} {case['id']:<22} {latency:5.1f}s intent={row['intent']:<10} "
              f"figures={g_ok}/{g_total} judge={row['faithfulness']} "
              + (f"failed={failed} " if failed else "") + (f"unsupported={unsupported}" if unsupported else ""))
        time.sleep(PAUSE_BETWEEN_CASES_S)

    return report(rows)


def _rate(rows, prefix):
    vals = [v for r in rows for k, v in r["checks"].items() if k == prefix or k.startswith(prefix + ":")]
    return (sum(vals) / len(vals)) if vals else 1.0, len(vals)


def report(rows: list[dict]) -> int:
    g_ok = sum(r["grounding"][0] for r in rows)
    g_tot = sum(r["grounding"][1] for r in rows)
    judged = [r["faithfulness"] for r in rows if r["faithfulness"] is not None]
    lat = sorted(r["latency_s"] for r in rows)
    metrics = {
        "routing": _rate(rows, "routing"),
        "parameters": _rate(rows, "param"),
        "behaviour": _rate(rows, "behaviour"),
        "grounding": ((g_ok / g_tot) if g_tot else 1.0, g_tot),
        "faithfulness": ((statistics.mean(judged)) if judged else 5.0, len(judged)),
    }

    print("\n" + "=" * 72)
    print(f"{'metric':<14}{'score':>10}{'n':>6}{'gate':>10}   result")
    print("-" * 72)
    all_pass = True
    for name, (score, n) in metrics.items():
        gate = GATES[name]
        ok = score >= gate
        all_pass &= ok
        shown = f"{score:.2f}" if name == "faithfulness" else f"{score:.1%}"
        gate_shown = f"{gate:.1f}" if name == "faithfulness" else f"{gate:.0%}"
        print(f"{name:<14}{shown:>10}{n:>6}{gate_shown:>10}   {'PASS' if ok else 'FAIL'}")
    print("-" * 72)
    print(f"latency p50 {statistics.median(lat):.1f}s  p95 {lat[int(0.95 * (len(lat) - 1))]:.1f}s  "
          f"cases {len(rows)}")
    by_cat = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(all(r["checks"].values()) and not r["unsupported_figures"])
    print("by category: " + ", ".join(f"{c} {sum(v)}/{len(v)}" for c, v in by_cat.items()))
    print("=" * 72)

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    payload = {"run_at": datetime.now().isoformat(timespec="seconds"),
               "metrics": {k: {"score": v[0], "n": v[1], "gate": GATES[k]} for k, v in metrics.items()},
               "passed": all_pass, "cases": rows}
    (out_dir / "latest.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"results written to {out_dir / 'latest.json'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", help="comma-separated case ids")
    parser.add_argument("--no-judge", action="store_true")
    args = parser.parse_args()
    sys.exit(run(set(args.only.split(",")) if args.only else None, not args.no_judge))
