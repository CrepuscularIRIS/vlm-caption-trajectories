"""Cheap checks requested by the rival review (existing data only, text output)."""
import collections, json, os, re, sys  # [scrub] os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))  # [scrub]
import numpy as np
from traj_metrics import features, recovery
P = Path(os.environ["PROJECT_DIR"])  # [scrub] expects outputs/ and evidence/ under it
EV = P / "evidence/v3_behavior"
load = lambda p: [json.loads(l) for l in open(p) if l.strip()]

print("## 1 val keys in training logs (last val block)")
for tag in ("l025", "l050"):
    txt = open(P / f"outputs/grpo_v3_A_floor_{tag}.log", errors="ignore").read()
    i = txt.rfind("val:")
    print(tag, re.sub(r"\x1b\[[0-9;]*m|\(Runner pid=\d+\)", "", txt[i:i + 260]).replace("\n", " | "))
import pyarrow.parquet as pq
print("val parquet rows:", pq.read_table(os.environ["DATA_DIR"] + "/grpo_caption_v51_v3/val.parquet").num_rows)

print("\n## 2 E2 paired re-crop increment and false_confirm")
for tag in ("A_rl", "A_fl025", "A_fl050"):
    base = {(r["uid"], r["sample_idx"]): r for r in load(EV / f"traj_{tag}.jsonl")}
    pert = load(EV / f"traj_{tag}_perturb.jsonl")
    inc, fc, btt_given = [], [], []
    for r in pert:
        b = base.get((r["uid"], r["sample_idx"]))
        f = recovery(r, b)
        if f is None or b is None:
            continue
        inc.append(f["recrop"] - int(b["n_crops"] >= 2))
        fc.append(f["false_confirm"])
        if f["recrop"]:
            btt_given.append(f["back_to_target"])
    base_multi = np.mean([int(b["n_crops"] >= 2) for b in base.values()])
    print(f"{tag}: covered {len(inc)}, base P(>=2 crops) {base_multi:.3f}, paired recrop increment {np.mean(inc):.3f}, "
          f"false_confirm {np.mean(fc):.3f}, back_to_target|recrop {np.mean(btt_given):.3f}")

print("\n## 3 redundant crops conditioned on >=2 crops")
from compare_cropcost import redundant
for tag in ("A_sft", "A_rl", "A_fl025", "A_fl050"):
    rs = load(EV / f"traj_{tag}.jsonl")
    multi = [r for r in rs if r["n_crops"] >= 2]
    hits = [redundant(r) for r in rs]
    print(f"{tag}: >=2 crops {len(multi)/len(rs):.3f}, red_any|>=2 {np.mean([redundant(r)[0] > 0 for r in multi]):.3f}, "
          f"red per crop {sum(h for h, _ in hits)/max(1, sum(n for _, n in hits)):.3f}, tokens per turn "
          f"{np.mean([r['gen_tokens']/(r['n_crops']+1) for r in rs]):.1f}")

print("\n## 4 training groups: floored mixed groups and advantage mass on direct samples (recomputed)")
for tag, lam in (("l025", .025), ("l050", .05)):
    rows = load(P / f"outputs/grpo_v3_A_floor_{tag}_rollouts.jsonl")
    g = collections.defaultdict(list)
    for r in rows:
        g[(r["step"], r["uid"])].append(r)
    cnt = collections.Counter(); mass = collections.defaultdict(float); cov = collections.defaultdict(float)
    for v in g.values():
        s = np.array([x["score"] for x in v]); c = np.array([x["correct"] for x in v]); k = np.array([x["crops"] or 0 for x in v])
        typ = "all_correct" if c.all() else ("all_wrong" if not c.any() else "mixed")
        std = s.std(ddof=1); a = (s - s.mean()) / max(std, .25)
        if typ == "mixed" and std < .25:
            cnt["mixed_floored"] += 1
        cnt[typ] += 1
        mass[typ] += a[k == 0].sum()
        cov[typ] += float(((a - a.mean()) * (k - k.mean())).sum())
    print(tag, dict(cnt), "| direct-sample advantage sum:", {t: round(m, 1) for t, m in mass.items()},
          "| sum A*(k-mean k):", {t: round(x, 1) for t, x in cov.items()})

print("\n## 5 per-question delta acc vs delta crops (fl050 - rl) by layer")
def perq(tag):
    d = collections.defaultdict(list)
    for r in load(EV / f"traj_{tag}.jsonl"):
        d[r["uid"]].append(r)
    return {u: (np.mean([x["ok"] for x in v]), np.mean([x["n_crops"] for x in v]), v[0]["layer"]) for u, v in d.items()}
rl, fl = perq("A_rl"), perq("A_fl050")
da = np.array([fl[u][0] - rl[u][0] for u in rl]); dk = np.array([fl[u][1] - rl[u][1] for u in rl])
print("corr(dacc, dcrops) =", round(float(np.corrcoef(da, dk)[0, 1]), 3))
q = np.quantile(dk, [1/3, 2/3])
for name, m in (("crop cut most", dk <= q[0]), ("middle", (dk > q[0]) & (dk <= q[1])), ("crop cut least", dk > q[1])):
    print(f"{name}: n={m.sum()}, mean dcrops {dk[m].mean():.2f}, mean dacc {da[m].mean():+.3f}")
