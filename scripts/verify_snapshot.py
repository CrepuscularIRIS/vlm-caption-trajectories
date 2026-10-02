#!/usr/bin/env python3
"""Verify the published metadata snapshot, not visual truth or training gains.

Standard library only. Does not import API adapters, access the network, or train.
"""
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COVER = "obs/runs/v51_cover6_20261002"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def summarize(p):
    selected = [r for r in p["selection"] if r["exportable"]]
    keys = [(e["item_id"], e["generator"]) for e in p["episodes"]]
    require(len(keys) == len(set(keys)), "Ambiguous episode keys in this snapshot")
    by_key = dict(zip(keys, p["episodes"]))
    chosen = [by_key[(r["item_id"], r["chosen"])] for r in selected]
    cover = [e for e in p["episodes"] if e["run"] == COVER]
    coverage = {k: sum(e["features"][k] for e in chosen)
                for k in ("direct", "hold", "revise", "fallback")}
    return {
        "items": len(p["selection"]),
        "unique_root_groups": len({e["group_id"] for e in p["episodes"]}),
        "generated_episodes": len(p["episodes"]),
        "auto_candidates": sum(r["auto_passed"] for r in p["selection"]),
        "exportable": len(selected),
        "export_root_groups": len({e["group_id"] for e in chosen}),
        "selected_with_crop": sum(e["crops"] > 0 for e in chosen),
        "coverage": coverage,
        "selected_by_generator": dict(Counter(r["chosen"] for r in selected)),
        "cover": {
            "items": len({e["item_id"] for e in cover}),
            "episodes": len(cover),
            "correct": sum(e["score"] == 1 for e in cover),
            "direct": sum(e["features"]["direct"] for e in cover),
            "with_crop": sum(e["crops"] > 0 for e in cover),
            "rollout_steps": sum(e["steps"] for e in cover),
            "format_retries": sum(e["format_retries"] for e in cover),
            "request_receipts": sum(n for k, n in p["call_counts"].items() if k.startswith(COVER + "/")),
            "hold": sum(e["features"]["hold"] for e in cover),
            "revise": sum(e["features"]["revise"] for e in cover),
            "fallback": sum(e["features"]["fallback"] for e in cover),
        },
        "lf_report": {k: p["lf_check"][k] for k in ("samples", "passed", "max_tokens", "mean_tokens", "cutoff")},
        "behavior_gate_passed": all(coverage[k] >= 2 for k in ("hold", "revise", "fallback")),
    }


def check(p, summary):
    selected = [r for r in p["selection"] if r["exportable"]]
    require(len({r["item_id"] for r in p["selection"]}) == len(p["selection"]), "Duplicate selection item")
    selected_keys = [(r["item_id"], r["chosen"]) for r in selected]
    export_keys = [(r["item_id"], r["generator"]) for r in p["export_records"]]
    require(selected_keys == export_keys, "Selection/export order or identity mismatch")
    require(all(r["manual"] == "KEEP" and r["pool"] == "clean_sft" for r in selected), "Export lacks KEEP")
    require(summary["export_root_groups"] == summary["exportable"], "Repeated source groups in export")
    require(all(e["split"] == "train" for e in p["episodes"]), "Unexpected split in snapshot")
    for e in p["episodes"]:
        require(e["features"]["direct"] == (e["crops"] == 0), "direct/crop mismatch")
        require(0 <= e["crops"] <= e["budget"] <= 8, "Invalid crop budget")
        for k in ("hold", "revise", "fallback"):
            require(e["features"][k] == (e["features"][k + "_events"] > 0), "Event/feature mismatch")
    reported = p["reported_summary"]
    require(summary["items"] == reported["items"], "Item count mismatch")
    require(summary["auto_candidates"] == reported["funnel"]["auto_candidates"], "Auto funnel mismatch")
    require(summary["exportable"] == reported["funnel"]["exportable"], "Export funnel mismatch")
    for k, n in summary["coverage"].items():
        require(n == reported["smoke_gate"][k], "Coverage mismatch: " + k)
    c = summary["cover"]
    require(c["request_receipts"] == c["rollout_steps"] + c["format_retries"], "Cover receipt mismatch")
    require({x["item_id"] for x in p["cover_manifest"]["items"]}
            == {x["item_id"] for x in p["episodes"] if x["run"] == COVER}, "Cover manifest mismatch")
    rows = p["lf_check"]["rows"]
    require(len(rows) == len(p["export_records"]) == p["lf_check"]["samples"], "LF row count mismatch")
    passed = 0
    for i, (e, row) in enumerate(zip(p["export_records"], rows)):
        require(row["idx"] == i, "LF ordering mismatch")
        require(e["assistant_turns"] == row["gpt_turns"], "LF assistant count mismatch")
        require(e["images"] == row["images"], "LF image count mismatch")
        passed += (row["gpt_turns"] == row["think_in_target"] == row["actions_in_target"]
                   and row["images"] == row["vision_blocks"]
                   and not row["receipt_leak"] and not row["truncated"])
    require(passed == p["lf_check"]["passed"], "LF pass total mismatch")
    require(max(r["tokens"] for r in rows) == p["lf_check"]["max_tokens"], "LF max mismatch")
    require(round(sum(r["tokens"] for r in rows) / len(rows)) == p["lf_check"]["mean_tokens"], "LF mean mismatch")

    episodes = {(e["item_id"], e["generator"]): e for e in p["episodes"]}
    for path in sorted((ROOT / "cases").glob("*.json")):
        case = read(path)
        ep = episodes[(case["item_id"], case["generator"])]
        require(case["score"]["score"] == ep["score"], "Case score mismatch")
        require(len(case["steps"]) == ep["steps"], "Case steps mismatch")
        require(len(case["views"]) == ep["crops"] + 1, "Case view count mismatch")
        views = {v["view_id"]: v for v in case["views"]}
        available = ["original_image"]
        require(views["original_image"]["parent"] is None, "Invalid root view")
        for i, step in enumerate(case["steps"], 1):
            require(step["step"] == i and step["available_before"] == available, "Case prefix mismatch")
            new = step["observation"]
            if new:
                require(new not in available and new in views, "Invalid returned view")
                require(views[new]["parent"] in available, "Future parent view")
                available.append(new)
        require(set(available) == set(views), "Unreached case view")
        require(all(v["asset_included"] is False and len(v["sha256"]) == 64 for v in views.values()), "View metadata mismatch")

    for rel, expected in read(ROOT / "manifest.sha256.json").items():
        path = (ROOT / rel).resolve()
        require(path.is_relative_to(ROOT), "Unsafe manifest path")
        require(hashlib.sha256(path.read_bytes()).hexdigest() == expected, "File hash changed: " + rel)


def main():
    p = read(ROOT / "data/snapshot.json")
    summary = summarize(p)
    require(summary == read(ROOT / "data/expected_summary.json"), "Frozen summary differs from recomputed metadata")
    check(p, summary)
    print(json.dumps({"consistency_check": "PASS", **summary}, ensure_ascii=False, indent=2))
    print("PASS = published artifact consistency only. Behavior coverage remains FAILED.")
    print("Images, private API receipts, full token masks and training gains are not verified here.")


if __name__ == "__main__":
    main()
