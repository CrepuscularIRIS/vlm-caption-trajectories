"""Null calibration of eps (CounterCredit App. C): leave-one-out over the K random patches.

Each random-patch score stands in for the real observation and is contrasted with the mean of the
other patches (zero expected evidence); the call's own D is kept. A null draw is falsely verified
when min(E_null, D) > eps. Also reports the real-call region shares at each eps.
Usage: python calibrate_eps.py <rollouts.jsonl> [more.jsonl ...]
"""
import json
import sys

GRID = [0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3]


def main(paths):
    calls = []
    for p in paths:
        for line in open(p):
            r = json.loads(line)
            for v in r.get("probe") or []:
                if v.get("eligible") and len(v.get("u_rand", [])) >= 2:
                    calls.append(v)
    nulls = []
    for v in calls:
        d = v["u_real"] - v["u_now"]
        rs = v["u_rand"]
        for k in range(len(rs)):
            others = rs[:k] + rs[k + 1:]
            nulls.append(min(rs[k] - sum(others) / len(others), d))
    print(f"eligible calls={len(calls)} null draws={len(nulls)}")
    if not calls:
        return
    print("eps   false_verify%  real_verified%  needed(D>eps)%  used(E>eps)%")
    for e in GRID:
        fv = 100 * sum(x > e for x in nulls) / len(nulls)
        rv = 100 * sum(v["q"] > e for v in calls) / len(calls)
        nd = 100 * sum(v["D"] > e for v in calls) / len(calls)
        us = 100 * sum(v["E"] > e for v in calls) / len(calls)
        print(f"{e:<5} {fv:13.2f} {rv:15.2f} {nd:15.2f} {us:13.2f}")
    first = next((e for e in GRID if sum(x > e for x in nulls) / len(nulls) < 0.03), None)
    print("first eps with false-verify < 3%:", first)
    hi = sum(v["u_now"] > 0.9 for v in calls) / len(calls)
    print(f"share of calls with U_now > 0.9 (answer already confident before crop): {100*hi:.1f}%")


if __name__ == "__main__":
    main(sys.argv[1:])
