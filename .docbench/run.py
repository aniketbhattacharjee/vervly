#!/usr/bin/env python
"""Document-level bench: does the joined document read as "AI, rewritten"
when its paragraphs read as human one at a time, and does document-level
selection (PipelineConfig.document_select_scans) change that?

    .venv/bin/python .docbench/run.py --scans 0 --tag perpara
    .venv/bin/python .docbench/run.py --scans 24 --tag docselect

Documents: the 9 AI bench paragraphs (.kriukow/gptzero_texts.json) joined
three at a time into three ~430-word essays, with their author facts.
Generation: the production Modal server (deploy/.env). Candidate scorer: the
surrogate, as in production. Verdict: GPTZero when GPTZERO_API_KEY answers,
otherwise the surrogate (then the document verdict means nothing about the
paraphrase class, and the results file says so).

Per document the file records the assembled verdict (label, subclass,
probability), the per-paragraph GPTZero verdicts of the shipped paragraphs,
the document_select block and timings. Results: .docbench/results/<tag>.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def load_env() -> None:
    env = ROOT / "deploy" / ".env"
    for line in env.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            if k in ("HUMANIZER_LLM_URL", "HUMANIZER_LLM_API_KEY") and not os.environ.get(k):
                os.environ[k] = v.strip()
    os.environ.setdefault("HUMANIZER_PROXY", "surrogate")
    os.environ.setdefault("HUMANIZER_ASSUME_AI", "1")
    os.environ.setdefault("HUMANIZER_LLM_TIMEOUT_S", "900")
    os.environ["HF_HUB_OFFLINE"] = "1"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scans", type=int, default=0, help="document_select_scans")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n-candidates", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="only the first N documents")
    args = ap.parse_args()
    load_env()
    sys.path.insert(0, str(ROOT / "src"))

    from humanizer.detectors.gptzero import GPTZeroClient
    from humanizer.humanize import pipeline as pipe

    texts = json.loads((ROOT / ".kriukow" / "gptzero_texts.json").read_text())
    facts = json.loads((ROOT / ".kriukow" / "gptzero_facts.json").read_text())
    ai = [t for t in texts if t["name"].startswith("ai")]
    docs = []
    for i in range(0, len(ai) - 2, 3):
        group = ai[i : i + 3]
        docs.append({
            "name": "+".join(g["name"] for g in group),
            "text": "\n\n".join(g["text"].strip() for g in group),
            "facts": "\n".join(facts.get(g["name"], "") for g in group).strip(),
        })
    if args.limit:
        docs = docs[: args.limit]

    gz = GPTZeroClient()
    gz_ok = True
    try:
        gz.score(
            "The committee met on Tuesday and agreed to postpone the vote until the "
            "budget figures were available to every member. Two members asked for the "
            "minutes of the previous meeting to be circulated again, and the chair agreed "
            "to send them by Friday along with the revised agenda for the next session."
        )
    except Exception as exc:  # noqa: BLE001
        gz_ok = False
        print("GPTZero not answering (%s): verdicts below come from the surrogate" % str(exc)[:80])
    os.environ["HUMANIZER_YARDSTICK"] = "gptzero" if gz_ok else "surrogate"

    out = {"tag": args.tag, "scans": args.scans, "gptzero": gz_ok, "docs": []}
    path = HERE / "results" / ("%s.json" % args.tag)
    for d in docs:
        cfg = pipe.PipelineConfig(
            n_candidates=args.n_candidates, rounds=args.rounds, facts=d["facts"],
            document_select_scans=args.scans,
        )
        t0 = time.time()
        result = None
        for ev in pipe.stream(d["text"], config=cfg):
            if ev.result_object is not None:
                result = ev.result_object
            if ev.stage == "finalize" and ev.status == "progress":
                print("   ", ev.detail)
        row = {
            "name": d["name"],
            "words": len(d["text"].split()),
            "seconds": round(time.time() - t0, 1),
            "after": {k: result.after.get(k) for k in ("ai_probability", "label", "subclass")},
            "document_select": result.summary["document_select"],
            "paragraphs": [],
            "humanized": result.humanized,
        }
        if gz_ok:
            for p in result.paragraphs:
                r = gz.score(p.chosen)
                sub = (r.raw.get("subclass") or {})
                sub = next((b.get("predicted_class") for b in sub.values() if isinstance(b, dict)), None)
                row["paragraphs"].append({"label": r.label, "p": round(r.ai_probability, 3), "subclass": sub})
        out["docs"].append(row)
        print("%-60s %4d words %6.1fs  doc: %s %s p=%.3f  paras: %s" % (
            row["name"], row["words"], row["seconds"], row["after"]["label"], row["after"]["subclass"] or "",
            row["after"]["ai_probability"] or -1, " ".join("%s/%s" % (q["label"], q["subclass"] or "-") for q in row["paragraphs"]) or "(no GPTZero)"))
        path.write_text(json.dumps(out, indent=1))
    print("wrote", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
