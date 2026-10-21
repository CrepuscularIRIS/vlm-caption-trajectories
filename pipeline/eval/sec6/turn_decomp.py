"""Turn-level decomposition of the dV and orig GRPO arms (direction A, docs/29 section 6.4). CPU only, reads existing
files, writes one JSON summary (no question text).

Training rollouts (outputs/grpo_v3_A_{dv_s42,floor_orig_s42}_rollouts.jsonl; one line per rollout, 16 per (step, uid)
group). Advantage is recomputed as EasyR1 GRPO does for these runs: A = (r - mean) / max(std_unbiased, 0.25), r = score.
Per arm and 40-step window:
  groups      share of all-correct / all-wrong / mixed groups
  mixed.*     in mixed groups only:
    crops_correct / crops_wrong     mean crops of correct vs wrong rollouts
    acc_by_crops                    accuracy by crop bucket (0, 1, 2, 3, 4+)
    seq_push                        mean over groups of mean_i(A_i * crops_i)  (= within-group cov(A, crops); > 0 means
                                    sequence-level advantage favours more crops)
    tok_push                        sum A_i * tok_i * crops_i / sum |A_i| * tok_i (token-averaged loss weights each
                                    rollout by its generated tokens; > 0 means the token-weighted update favours more crops)
    tok_mass_by_crops               sum A_i * tok_i per crop bucket / sum |A_i| * tok_i (net signed token mass)
    corr_tok_A                      mean within-group Pearson corr(gen_tokens, A)
    first_crop                      first-turn crop share (dv log field 'first'; orig log has none -> crops > 0 proxy)
  dv_term (dv arm only)             group mean of s*dV in mixed vs pure (all-correct / all-wrong) groups; in pure groups
                                    the outcome advantage is 0, so this term is the only update on first-turn tokens
E1 trajectories (evidence/dvgrpo_20261020/traj_{arm}{step}.jsonl, 552 items x 4):
  crops_dist, acc_by_crops, and per crop index k (1..8): count, action word share (Step N | WORD), source share
  (original_image vs observation_j; behaviour word from parsed.behavior, else the Step line), relation to earlier crops (repeat: IoU >= 0.7 with an earlier crop box;
  inside: >= 90% of its area inside the previous crop box; new: otherwise), mean tokens of the turn issuing it.
  Also tokens after the first crop and the share of items whose 4 samples all end with the same letter.
Usage: python turn_decomp.py <outputs_dir> <eval_dir> <out.json>
"""
import collections
import json
import re
import statistics
import sys

ARMS = {"dv": "grpo_v3_A_dv_s42", "orig": "grpo_v3_A_floor_orig_s42"}
STEPS = (40, 80, 120, 160)
WORD = re.compile(r"Step\s*\d+\s*\|\s*([A-Za-z_]+)")


def bucket(c: int) -> str:
    return str(c) if c < 4 else "4+"


def window(step: int) -> str:
    lo = (step - 1) // 40 * 40 + 1
    return f"s{lo}-{lo + 39}"


def advantages(rs: list) -> list:
    m = sum(rs) / len(rs)
    sd = statistics.stdev(rs) if len(rs) > 1 else 0.0
    return [(r - m) / max(sd, 0.25) for r in rs]


def pearson(x: list, y: list) -> float | None:
    if len(set(x)) < 2 or len(set(y)) < 2:
        return None
    mx, my = sum(x) / len(x), sum(y) / len(y)
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
    vx = sum((a - mx) ** 2 for a in x) ** 0.5
    vy = sum((b - my) ** 2 for b in y) ** 0.5
    return cov / (vx * vy)


def train_stats(path: str) -> dict:
    groups = collections.defaultdict(list)
    for line in open(path):
        r = json.loads(line)
        groups[(r["step"], r["uid"])].append(r)
    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    for (step, _), g in sorted(groups.items()):
        w = window(step)
        rs = [float(x["score"]) for x in g]
        ok = [float(x["correct"]) for x in g]
        kind = "all_correct" if min(ok) == 1 else "all_wrong" if max(ok) == 0 else "mixed"
        acc[w]["kind"].append(kind)
        if "dv" in g[0]:  # dv arm only: the first-turn term s*dV is the only gradient in non-mixed groups
            term = [(1.0 if x.get("first") == "grounding" else -1.0 if x.get("first") == "answer" else 0.0) * float(x["dv"]) for x in g]
            acc[w]["term_" + ("mixed" if kind == "mixed" else "pure")].append(sum(term) / len(term))
        if kind != "mixed":
            continue
        A = advantages(rs)
        cr = [int(x["crops"]) for x in g]
        tk = [int(x["gen_tokens"]) for x in g]
        d = acc[w]
        d["crops_c"] += [c for c, o in zip(cr, ok) if o == 1]
        d["crops_w"] += [c for c, o in zip(cr, ok) if o == 0]
        for c, o in zip(cr, ok):
            d["acc_" + bucket(c)].append(o)
        d["seq_push"].append(sum(a * c for a, c in zip(A, cr)) / len(A))
        d["tp_num"].append(sum(a * t * c for a, t, c in zip(A, tk, cr)))
        d["tp_den"].append(sum(abs(a) * t for a, t in zip(A, tk)))
        for a, t, c in zip(A, tk, cr):
            d["mass_" + bucket(c)].append(a * t)
        p = pearson(tk, A)
        if p is not None:
            d["corr_tok_A"].append(p)
        d["first_crop"] += [1.0 if (x.get("first") == "grounding" if "first" in x else int(x["crops"]) > 0) else 0.0 for x in g]
    out = {}
    for w, d in acc.items():
        n = len(d["kind"])
        den = sum(d["tp_den"]) or 1.0
        mean = lambda v: sum(v) / len(v) if v else None  # noqa: E731
        out[w] = dict(
            n_groups=n,
            groups={k: d["kind"].count(k) / n for k in ("all_correct", "all_wrong", "mixed")},
            mixed=dict(
                crops_correct=mean(d["crops_c"]), crops_wrong=mean(d["crops_w"]),
                acc_by_crops={b: dict(n=len(d["acc_" + b]), acc=mean(d["acc_" + b])) for b in ("0", "1", "2", "3", "4+")},
                seq_push=mean(d["seq_push"]), tok_push=sum(d["tp_num"]) / den,
                tok_mass_by_crops={b: sum(d["mass_" + b]) / den for b in ("0", "1", "2", "3", "4+")},
                corr_tok_A=mean(d["corr_tok_A"]), first_crop=mean(d["first_crop"])),
            dv_term=dict(mean_in_mixed=mean(d["term_mixed"]), mean_in_pure=mean(d["term_pure"]),
                         n_pure_groups_with_term=sum(abs(v) > 0 for v in d["term_pure"])) if d["term_pure"] or d["term_mixed"] else None)
    return out


def iou(a: list, b: list) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def inside(a: list, b: list) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area = (a[2] - a[0]) * (a[3] - a[1])
    return ix * iy / area if area > 0 else 0.0


def lit(s):
    import ast
    return ast.literal_eval(s) if isinstance(s, str) else s


def eval_stats(path: str) -> dict:
    rows = [json.loads(l) for l in open(path)]
    crops = collections.Counter()
    accb = collections.defaultdict(list)
    per_k = collections.defaultdict(lambda: collections.defaultdict(int))
    tok_k = collections.defaultdict(list)
    tok_after_first = []
    letters = collections.defaultdict(set)
    for r in rows:
        c = int(r["n_crops"])
        ok = float(r["ok"])
        crops[bucket(c)] += 1
        accb[bucket(c)].append(ok)
        letters[r["uid"]].add(r.get("pred"))
        turns = lit(r["turns"])
        crop_turns = [t for t in turns if (t.get("parsed") or {}).get("kind") == "grounding"]
        for k, t in enumerate(crop_turns, start=1):
            beh = (t.get("parsed") or {}).get("behavior")
            if not beh:
                m = WORD.search(t.get("text", ""))
                beh = m.group(1).lower() if m else "none"
            pk = per_k[k]
            pk["n"] += 1
            pk["word_" + beh] += 1
            pk["src_" + ("original" if (t.get("parsed") or {}).get("source") == "original_image" else "observation")] += 1
            tok_k[k].append(int(t.get("n_tokens", 0)))
            boxes = [x.get("box_abs") for x in crop_turns[:k]]
            if k >= 2 and all(boxes):
                b, prev = boxes[-1], boxes[:-1]
                if any(iou(b, p) >= 0.7 for p in prev):
                    pk["rel_repeat"] += 1
                elif inside(b, prev[-1]) >= 0.9:
                    pk["rel_inside"] += 1
                else:
                    pk["rel_new"] += 1
        if crop_turns:
            first_round = crop_turns[0].get("round", 0)
            after = sum(int(t.get("n_tokens", 0)) for t in turns if t.get("round", 0) > first_round)
            tok_after_first.append(after)
    n = len(rows)
    out = dict(n=n, crops_dist={b: crops[b] / n for b in ("0", "1", "2", "3", "4+")},
               acc_by_crops={b: dict(n=len(v), acc=sum(v) / len(v)) for b, v in sorted(accb.items())},
               tok_after_first=sum(tok_after_first) / max(1, len(tok_after_first)),
               same_letter_all4=sum(len(v) == 1 for v in letters.values()) / max(1, len(letters)),
               per_k={})
    for k in sorted(per_k):
        pk = per_k[k]
        cnt = pk["n"]
        out["per_k"][k] = dict(n=cnt, per_item=cnt / n, tok=sum(tok_k[k]) / cnt,
                               **{f: v / cnt for f, v in sorted(pk.items()) if f != "n"})
    return out


def main() -> None:
    outdir, evdir, dst = sys.argv[1:4]
    res = dict(train={a: train_stats(f"{outdir}/{p}_rollouts.jsonl") for a, p in ARMS.items()},
               e1={a: {s: eval_stats(f"{evdir}/traj_{a}{s}.jsonl") for s in STEPS} for a in ARMS})
    json.dump(res, open(dst, "w"), indent=1)
    print("wrote", dst)


if __name__ == "__main__":
    main()
