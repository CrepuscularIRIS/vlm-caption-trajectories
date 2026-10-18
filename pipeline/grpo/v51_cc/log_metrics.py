"""Turn EasyR1 console step blocks ("Step N" + indented metric tree) into one CSV row per step.

Usage: python log_metrics.py RUN.log > RUN_metrics.csv
Only (Runner ...) lines are read; nested keys are joined with "/". The initial validation
("Step 0") is step 0. [counter_credit] price lines are attached to the step
whose metrics block follows them.
"""
import csv, re, sys

ANSI = re.compile(r"\x1b\[[0-9;]*m")
STEP = re.compile(r"^Step (\d+)$")
KV = re.compile(r"^( *)([A-Za-z0-9_@/\-.]+):\s*(.*)$")
PRICE = re.compile(r"\[counter_credit\] price_mean=([-0-9.e]+) price_adv_abs_mean=([-0-9.e]+) verified=(\d+) crops=(\d+)")

def main(path):
    rows, cur, stack, pending = [], {"step": 0}, [], {}
    for raw in open(path, errors="replace"):
        line = ANSI.sub("", raw.rstrip("\n"))
        p = PRICE.search(line)
        if p and "Runner" in line:
            pending = dict(zip(("cc/price_mean", "cc/price_adv_abs_mean", "cc/verified", "cc/crops"), p.groups()))
            continue
        if "(Runner pid=" not in line:
            continue
        body = re.sub(r"^\d+\) ?", "", line.rsplit("(Runner pid=", 1)[1])
        s = STEP.match(body.strip())
        if s:
            if len(cur) > 1:
                rows.append(cur)
            cur, stack = {"step": int(s.group(1)), **pending}, []
            pending = {}
            continue
        kv = KV.match(body)
        if not kv:
            continue
        depth = len(kv.group(1)) // 2
        stack = stack[:depth] + [kv.group(2)]
        if kv.group(3) and re.fullmatch(r"[-0-9.eE+]+|nan|inf", kv.group(3)):
            cur["/".join(stack)] = kv.group(3)
    if len(cur) > 1:
        rows.append(cur)
    rows = [r for r in rows if any(k.startswith(("val", "reward", "actor", "critic")) for k in r)]
    merged = {}
    for r in rows:  # the last step is printed again after the final validation; merge by step
        merged.setdefault(r["step"], {}).update(r)
    rows = [merged[k] for k in sorted(merged)]
    keys = ["step"] + sorted({k for r in rows for k in r} - {"step"})
    w = csv.DictWriter(sys.stdout, keys)
    w.writeheader()
    w.writerows(rows)

if __name__ == "__main__":
    main(sys.argv[1])
