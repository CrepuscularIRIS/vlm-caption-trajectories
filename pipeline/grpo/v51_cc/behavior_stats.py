"""Behaviour census: library-A SFT targets vs E1/E2 eval outputs of each model.
Per trajectory: n turns, behaviours (Step N | WORD), HOLD/REVISE update lines, UNCLEAR answers.
E2: whether the model re-cropped after the perturbed first crop, and accuracy then."""
import collections, json, re, sys
HDR = re.compile(r"Step\s+\d+\s*\|\s*([a-z_]+)", re.I)
UPD = re.compile(r"^\s*(HOLD|REVISE)\s*:", re.M)

def summarize(name, trajs):
    n = len(trajs); beh = collections.Counter(); upd = collections.Counter(); per = collections.Counter()
    for t in trajs:
        seen = set()
        for b in t["beh"]:
            beh[b] += 1; seen.add(b)
        for u in t["upd"]:
            upd[u] += 1; seen.add(u)
        if t.get("unclear"): seen.add("UNCLEAR")
        for s in seen: per[s] += 1
    keys = ["locate", "re_examine", "fallback", "commit", "abstain", "HOLD", "REVISE", "UNCLEAR"]
    print(f"| {name} | {n} | " + " | ".join(f"{per[k]/n:.3f}" for k in keys) + " |")

def sft(path):
    out = []
    for x in json.load(open(path)):
        g = [c["value"] for c in x["conversations"] if c["from"] == "gpt"]
        out.append({"beh": [m.group(1).lower() for v in g for m in HDR.finditer(v)][:len(g)],
                    "upd": [m.group(1) for v in g for m in UPD.finditer(v)],
                    "unclear": "<answer>UNCLEAR</answer>" in g[-1]})
    return out

def ev(path):
    out = []
    for l in open(path):
        r = json.loads(l); T = r.get("turns") or []
        beh, upd = [], []
        for t in T:
            p = t.get("parsed") or {}
            if p.get("behavior"): beh.append(p["behavior"].lower())
            upd += [u.get("kind") for u in (p.get("updates") or [])]
        last = (T[-1].get("parsed") or {}) if T else {}
        out.append({"beh": beh, "upd": upd, "unclear": (last.get("answer") or "").upper() == "UNCLEAR", "r": r})
    return out

print("| source | n | locate | re_examine | fallback | commit | abstain | HOLD | REVISE | UNCLEAR |")
print("|---|---|---|---|---|---|---|---|---|---|")
summarize("library A SFT data", sft(sys.argv[1]))
for tag in sys.argv[2:]:
    summarize(tag, ev(f"traj_{tag}.jsonl"))
print()
print("| E2 model | n | re-cropped after perturbed crop | acc if re-cropped | acc if not | REVISE rate |")
print("|---|---|---|---|---|---|")
for tag in sys.argv[2:]:
    try: T = ev(f"traj_{tag}_perturb.jsonl")
    except FileNotFoundError: continue
    pert = [t for t in T if t["r"].get("perturb") not in (None, "none", False, {}) and t["r"]["n_crops"] >= 1]
    if not pert: continue
    rec = [t for t in pert if t["r"]["n_crops"] >= 2]; no = [t for t in pert if t["r"]["n_crops"] < 2]
    acc = lambda L: sum(t["r"]["ok"] for t in L) / len(L) if L else float("nan")
    print(f"| {tag} | {len(pert)} | {len(rec)/len(pert):.3f} | {acc(rec):.3f} | {acc(no):.3f} | {sum('REVISE' in t['upd'] for t in pert)/len(pert):.3f} |")
