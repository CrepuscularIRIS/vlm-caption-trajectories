"""Plain efficiency measure for '2 crops would do, the model used 4-5': on each question (several eval samples), the
fewest crops any correct sample used is the reference; every correct sample's excess = its crops - that minimum.
With 4 samples per question the minimum is an upper bound on what is needed, so the excess is a lower bound.

Usage: python3 excess_crops.py <out.json> <traj.jsonl> [<traj.jsonl> ...]
"""
import collections
import json
import statistics
import sys


def summary(rows: list) -> dict:
    by_q = collections.defaultdict(list)
    for r in rows:
        by_q[r["uid"]].append((int(r["n_crops"]), int(bool(r["ok"]))))
    samples = [s for v in by_q.values() for s in v]
    crops_all = sum(n for n, _ in samples)
    excess = []
    for v in by_q.values():
        ok = [n for n, y in v if y]
        excess += [n - min(ok) for n in ok] if ok else []
    dist = collections.Counter(min(n, 5) for n, _ in samples)
    return {
        "samples": len(samples), "questions": len(by_q),
        "accuracy": statistics.fmean(y for _, y in samples), "crops_per_sample": crops_all / len(samples),
        "crop_count_share(5=5+)": {str(k): v / len(samples) for k, v in sorted(dist.items())},
        "correct_mean_excess": statistics.fmean(excess),
        "correct_share_excess_ge2": sum(e >= 2 for e in excess) / len(excess),
        "excess_crops_share_of_all_crops": sum(excess) / crops_all,
        "crops_in_wrong_samples_share": sum(n for n, y in samples if not y) / crops_all,
    }


def main() -> None:
    out = {p.rsplit("/", 1)[-1]: summary([json.loads(x) for x in open(p) if x.strip()]) for p in sys.argv[2:]}
    with open(sys.argv[1], "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
