"""Decision-loss MVE comparison and pre-registered verdict (PREREG.md). Usage: dl_compare.py <evidence dir> <outputs dir> <out.json>"""
import json
import random
import re
import sys
from pathlib import Path

O, OUT, DEST = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
B, SEED = 10000, 0


def boot(xs):
    """Percentile bootstrap 95% CI of the mean over units (questions/states)."""
    if not xs:
        return None
    rng = random.Random(SEED)
    n = len(xs)
    means = sorted(sum(xs[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return [means[int(0.025 * B)], means[int(0.975 * B) - 1]]


def summary(xs):
    return {"n": len(xs), "mean": sum(xs) / len(xs) if xs else None, "ci95": boot(xs)}


def load(p):
    return json.load(open(O / p))


# ---- layer 2: choice utility on the common reference states
states = load("audit_states.json")["states"]
sc = {k: {s["uid"]: s for s in load(f"score_{k}.json")["scores"]} for k in ("ref", "main", "ctrl")}
u_terms, u_main_ref, u_ctrl_ref, per_state = [], [], [], []
for st in states:
    if not st["usable"] or "delta_audit" not in st:
        u_terms.append(0.0), u_main_ref.append(0.0), u_ctrl_ref.append(0.0)
        continue
    d = st["delta_audit"]
    pm, pc, pr = (sc[k][st["uid"]]["p_answer"] for k in ("main", "ctrl", "ref"))
    u_terms.append((pm - pc) * d)
    u_main_ref.append((pm - pr) * d)
    u_ctrl_ref.append((pc - pr) * d)
    per_state.append({"uid": st["uid"], "delta": d, "p_ref": pr, "p_main": pm, "p_ctrl": pc})
layer2 = {"U_choice": summary(u_terms), "U_main_vs_ref": summary(u_main_ref), "U_ctrl_vs_ref": summary(u_ctrl_ref),
          "usable_states": sum(s["usable"] for s in states), "states": len(states),
          "delta_sign": {"pos": sum(s.get("delta_audit", 0) > 0 for s in states),
                         "neg": sum(s.get("delta_audit", 0) < 0 for s in states),
                         "zero": sum(s["usable"] and s.get("delta_audit") == 0 for s in states)},
          "mean_p": {k: sum(x[f"p_{k}"] for x in per_state) / max(1, len(per_state)) for k in ("ref", "main", "ctrl")},
          "mean_mass": {k: sum(sc[k][x["uid"]]["mass"] for x in per_state) / max(1, len(per_state))
                        for k in ("ref", "main", "ctrl")}}


# ---- layer 3: clean / E2 on the 64 questions, paired by question
def per_question(path):
    rows = [json.loads(line) for line in open(path) if line.strip()]
    rows = [r for r in rows if "uid" in r]
    q = {}
    for r in rows:
        inval = int(bool(r.get("error")) or bool(r.get("overlong")) or
                    any(t.get("check_errors") for t in r.get("turns", [])))
        first_kind = (r["turns"][0]["parsed"].get("kind") if r.get("turns") else None)
        y = r["ok_train_rule"]
        q.setdefault(r["uid"], []).append({"y": y, "r": y * (1 - 0.01 * r["n_crops"]), "crops": r["n_crops"],
                                           "gen": r["gen_tokens"], "invalid": inval,
                                           "direct": int(first_kind == "answer"),
                                           "hit": int(bool(r.get("perturb") and r["perturb"].get("shown_bbox")))})
    return {u: {k: sum(s[k] for s in v) / len(v) for k in v[0]} for u, v in q.items()}


layer3 = {}
for cond, suf in (("clean", ""), ("e2", "_perturb")):
    m, c = per_question(O / f"traj_main{suf}.jsonl"), per_question(O / f"traj_ctrl{suf}.jsonl")
    common = sorted(set(m) & set(c))
    layer3[cond] = {"questions": len(common)}
    for k in ("y", "r", "crops", "gen", "invalid", "direct", "hit"):
        layer3[cond][k] = {"main": sum(m[u][k] for u in common) / len(common),
                           "ctrl": sum(c[u][k] for u in common) / len(common),
                           "diff": summary([m[u][k] - c[u][k] for u in common])}

# ---- recovery: same shifted-crop states, 2 continuations each
rec = {}
for k in ("main", "ctrl"):
    by = {}
    for r in load(f"recover_{k}.json")["recovery"]:
        by.setdefault(r["uid"], []).append(r["y"])
    rec[k] = {u: sum(v) / len(v) for u, v in by.items()}
rc = sorted(set(rec["main"]) & set(rec["ctrl"]))
recovery = {"states": len(rc), "acc_main": sum(rec["main"][u] for u in rc) / max(1, len(rc)),
            "acc_ctrl": sum(rec["ctrl"][u] for u in rc) / max(1, len(rc)),
            "diff": summary([rec["main"][u] - rec["ctrl"][u] for u in rc]),
            "missing": [s["uid"] for s in states[:16] if s.get("recovery", {}).get("status") != "ok"]}


# ---- layer 1 and cost from training logs
def train_log(tag):
    text = (OUT / f"grpo_v3_A_dl_{tag}.log").read_text(errors="ignore")
    t = [float(x) for x in re.findall(r"time_per_step: ([0-9.]+)", text)]
    forks = re.findall(r"\[dl_forks\] (.*)", text)
    aux = re.findall(r"\[dl_aux\] (.*)", text)
    return {"steps": len(t), "secs_per_step_mean": sum(t) / len(t) if t else None, "train_secs": sum(t),
            "gpu_hours_train": 2 * sum(t) / 3600, "dl_forks_lines": len(forks), "dl_aux_lines": len(aux),
            "dl_forks_first": forks[:1], "dl_forks_last": forks[-1:], "dl_aux_last": aux[-2:]}


train = {k: train_log(k) for k in ("main", "ctrl")}
fork_rows = [json.loads(line) for p in sorted(OUT.glob("grpo_v3_A_dl_main_forks.jsonl.*")) for line in open(p)
             if line.strip()]
fr = [r for r in fork_rows if "action" in r]
stt = [r for r in fork_rows if "state_status" in r]
train["main"]["forks"] = {"n": len(fr), "gen_tokens": sum(r["gen_tokens"] for r in fr),
                          "terminal": {t: sum(r["terminal"] == t for r in fr) for t in {r["terminal"] for r in fr}},
                          "states": len(stt), "usable": sum(r["state_status"] == "ok" for r in stt),
                          "q_answer": sum(r["r"] for r in fr if r["action"] == "answer") / max(1, sum(r["action"] == "answer" for r in fr)),
                          "q_crop": sum(r["r"] for r in fr if r["action"] == "grounding") / max(1, sum(r["action"] == "grounding" for r in fr))}

# ---- pre-registered verdict
u = layer2["U_choice"]["ci95"]
cl, e2, rv = layer3["clean"]["y"]["diff"], layer3["e2"]["y"]["diff"], recovery["diff"]
inv_gap = layer3["clean"]["invalid"]["main"] - layer3["clean"]["invalid"]["ctrl"]
harm = any(x["ci95"] is not None and x["ci95"][1] < 0 for x in (cl, e2, rv))
if u[1] < 0 or harm or inv_gap > 0.05:
    verdict = "refuted"
elif u[0] > 0 and cl["mean"] >= -0.02 and cl["ci95"][1] >= 0 and not harm and inv_gap <= 0.05:
    verdict = "supported"
else:
    verdict = "undecided"
res = {"verdict": verdict, "label": "development diagnostic (E1 subset)", "layer2": layer2, "layer3": layer3,
       "recovery": recovery, "train": train, "invalid_gap_clean": inv_gap, "per_state": per_state}
Path(DEST).write_text(json.dumps(res, indent=1))
print(json.dumps({"verdict": verdict, "U_choice": layer2["U_choice"], "clean_acc_diff": cl, "e2_acc_diff": e2,
                  "recovery_diff": rv, "crops_clean": layer3["clean"]["crops"], "direct_clean": layer3["clean"]["direct"],
                  "gpu_hours_train": {k: train[k]["gpu_hours_train"] for k in train}}, indent=1))
