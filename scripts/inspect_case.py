#!/usr/bin/env python3
"""Inspect real visible protocol text; no images or model calls are required."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/reference"))
import caption_v51 as protocol  # noqa: E402


def inspect(path):
    case = json.loads(path.read_text(encoding="utf-8"))
    snapshot = json.loads((ROOT / "data/snapshot.json").read_text(encoding="utf-8"))
    ep = next(e for e in snapshot["episodes"]
              if (e["item_id"], e["generator"]) == (case["item_id"], case["generator"]))
    rows, crops = [], 0
    for step in case["steps"]:
        p = protocol.parse_step(step["raw_text"])
        violations = protocol.check_step(p, step["step"], crops, step["available_before"], ep["budget"])
        if p.get("behavior") != step["behavior"]:
            violations.append("published_behavior_mismatch")
        if bool(step["observation"]) != (p.get("kind") == "grounding"):
            violations.append("action_observation_mismatch")
        rows.append({"step": step["step"], "behavior": p.get("behavior"),
                     "updates": [u["kind"] for u in p.get("updates", [])],
                     "violations": violations, "soft_flags": protocol.soft_flags(p)})
        crops += p.get("kind") == "grounding"
    ok = not any(r["violations"] for r in rows) and crops == ep["crops"]
    print(json.dumps({"case": path.name, "protocol_check": "PASS" if ok else "FAIL", "steps": rows},
                     ensure_ascii=False, indent=2))
    print("Protocol validity does not certify visual support, valid HOLD/REVISE, or clean-SFT eligibility.")
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path, nargs="?")
    parser.add_argument("--all", action="store_true", help="Inspect all five published real cases")
    args = parser.parse_args()
    if args.all and args.case or not args.all and args.case is None:
        parser.error("Provide one case file or --all")
    paths = sorted((ROOT / "cases").glob("*.json")) if args.all else [args.case]
    results = [inspect(path) for path in paths]
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
