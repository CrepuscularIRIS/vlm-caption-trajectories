"""Work package B (zero GPU): candidate states and a human image-review package for the paired-receipt MVE.

Candidates: SFT-A crop-branch trajectories on RL-pool questions (evidence/twobranch_20261019 + twobranch_pool_20261020,
the fixed-SFT-A two-branch runs). One focal crop per distinct question. Fixed design, seed 20261022:
  20 first-crop states (focal = crop 1) + 20 follow-up states (focal = crop k >= 2, k drawn uniformly among the
  trajectory's crops >= 2); within each, 12 whose old trajectory ended wrong and 8 that ended right; distinct uids.
For every candidate, the student's ACTUAL inputs are rebuilt with the crop environment and the training/eval image
processor (process_image, min 1024, max 589,824 px):
  views_before   every view the student had seen before the focal crop (processed), with receipts
  T              the processed real receipt of the focal crop (what the student received)
  C              matched control: the SAME absolute box cut from the student's already-shown PROCESSED source view
                 (the view the crop was taken from), resampled (bicubic) to T's exact size -> same region, size and
                 grid, no new detail
Writes review/<id>/{view_*.png, T.png, C.png}, review/index.html (one section per state: question + options,
history captions/actions up to the focal crop, images; the old continuation and final answer are behind a closed
<details> so the reviewer can label facts first), review/states.json (machine record, no images) and
review/annotation_template.csv (fields for the human labels).
Usage: python build_review.py <out_dir>"""
import html
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, os.environ["CAPTION_GRPO_CODE"])
sys.path.insert(0, os.environ["EASYR1_DIR"])
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402
from verl.utils.dataset import process_image  # noqa: E402

import caption_env as CENV  # noqa: E402

MIN_PX, MAX_PX = 1024, 589824
POOL = os.environ["DATA_DIR"] + "/grpo_caption_v51_v3"  # [scrub]
SRC = ["evidence/twobranch_20261019/traj_A_sft_crop.jsonl", "evidence/twobranch_pool_20261020/traj_A_sft_crop.jsonl"]
out = Path(sys.argv[1])
(out / "review").mkdir(parents=True, exist_ok=True)
df = pd.read_parquet(f"{POOL}/train.parquet")
ROWS = {r["uid"]: r for _, r in df.iterrows()}


def qtext(row):
    ex = row["extra_info"]
    if isinstance(ex, str):
        import ast
        ex = ast.literal_eval(ex)
    return ex["question"]


trajs = []
for p in SRC:
    for line in open(p):
        t = json.loads(line)
        if t.get("error") or t.get("overlong") or int(t["n_crops"]) < 1 or t["uid"] not in ROWS:
            continue
        trajs.append(t)
rng = random.Random(20261022)
rng.shuffle(trajs)
want = {("first", 0): 12, ("first", 1): 8, ("follow", 0): 12, ("follow", 1): 8}
got, used, picks = {k: 0 for k in want}, set(), []
for t in trajs:
    ok = int(t["ok"])
    kinds = ["first"] + (["follow"] if int(t["n_crops"]) >= 2 else [])
    rng.shuffle(kinds)
    for kind in kinds:
        if t["uid"] in used or got[(kind, ok)] >= want[(kind, ok)]:
            continue
        focal = 1 if kind == "first" else rng.randint(2, int(t["n_crops"]))
        picks.append((kind, ok, focal, t))
        got[(kind, ok)] += 1
        used.add(t["uid"])
        break
    if all(got[k] >= want[k] for k in want):
        break


def rebuild(t, focal):
    """Replay the environment up to and including the focal crop; return processed views and T/C."""
    row = ROWS[t["uid"]]
    env = CENV.CropEnv(row["images"][0])
    shown = [process_image(env.images[0], MIN_PX, MAX_PX)]
    crop_turns = [x for x in t["turns"] if (x.get("parsed") or {}).get("kind") == "grounding"]
    hist = []
    for k, turn in enumerate(crop_turns[:focal], start=1):
        p = dict(turn["parsed"])
        src = CENV.resolve_source(p.get("source", ""), len(env.boxes))
        r = env.step(p)
        if r is None or list(env.boxes[-1]) != list(turn["box_abs"]):
            return None
        hist.append(dict(k=k, text=turn.get("text_in_history", turn["text"]), source=p.get("source"),
                         bbox=p.get("bbox"), box_abs=list(env.boxes[-1]), receipt=r["head"] + " … " + r["tail"]))
        T = process_image(env.images[-1], MIN_PX, MAX_PX)
        if k < focal:
            shown.append(T)
            continue
        # matched control C: the same absolute box, cut from the processed source view the student already saw
        P = shown[src]
        sx0, sy0, sx1, sy1 = env.boxes[src]
        ax0, ay0, ax1, ay1 = env.boxes[-1]
        fx, fy = P.width / (sx1 - sx0), P.height / (sy1 - sy0)
        region = (max(0, (ax0 - sx0) * fx), max(0, (ay0 - sy0) * fy),
                  min(P.width, (ax1 - sx0) * fx), min(P.height, (ay1 - sy0) * fy))
        C = P.crop(tuple(round(v) for v in region)).resize(T.size, Image.BICUBIC)
        return dict(shown=shown, T=T, C=C, hist=hist, src=src, region_in_source_px=[round(v, 1) for v in region],
                    source_shown_wh=list(P.size), T_wh=list(T.size), raw_crop_wh=[round(ax1 - ax0), round(ay1 - ay0)])
    return None


states, sections = [], []
for n, (kind, ok, focal, t) in enumerate(picks):
    rb = rebuild(t, focal)
    sid = f"s{n:02d}"
    rec = dict(id=sid, uid=t["uid"], sample_idx=t["sample_idx"], source=t["source"], gt=t["gt"], kind=kind,
               focal_crop=focal, old_ok=ok, old_pred=t.get("pred"), n_crops=int(t["n_crops"]), rebuilt=rb is not None)
    if rb is None:
        states.append(rec)
        continue
    d = out / "review" / sid
    d.mkdir(exist_ok=True)
    for i, im in enumerate(rb["shown"]):
        im.save(d / f"view_{i}.png")
    rb["T"].save(d / "T.png")
    rb["C"].save(d / "C.png")
    rec.update(src_view=rb["src"], region_in_source_px=rb["region_in_source_px"], source_shown_wh=rb["source_shown_wh"],
               T_wh=rb["T_wh"], raw_crop_wh=rb["raw_crop_wh"],
               detail_gain=round((rb["raw_crop_wh"][0] * rb["raw_crop_wh"][1]) /
                                 max(1, (rb["region_in_source_px"][2] - rb["region_in_source_px"][0]) *
                                     (rb["region_in_source_px"][3] - rb["region_in_source_px"][1])), 2))
    states.append(rec)
    q = html.escape(qtext(ROWS[t["uid"]]))
    hist_html = "".join(f"<li><b>turn {h['k']}</b> source={html.escape(str(h['source']))} bbox={h['bbox']}<pre>"
                        f"{html.escape(h['text'])}</pre><small>{html.escape(h['receipt'][:300])}</small></li>"
                        for h in rb["hist"])
    later = t["turns"][focal:]  # turns 1..focal issued crops 1..focal; the rest is the old continuation
    later_html = "".join(f"<pre>{html.escape(x['text'])}</pre>" for x in later)
    views = "".join(f"<figure><img src='{sid}/view_{i}.png' style='max-width:480px'><figcaption>view {i} (as shown)"
                    f"</figcaption></figure>" for i in range(len(rb["shown"])))
    sections.append(f"""<section id='{sid}'><h2>{sid} · {kind} crop (focal crop {focal}) · {t['source']}</h2>
<p><b>Question</b><pre>{q}</pre></p><h3>Views the student had before the focal crop</h3>{views}
<h3>History up to and including the focal crop action</h3><ol>{hist_html}</ol>
<h3>Focal receipt</h3><figure><img src='{sid}/T.png' style='max-width:480px'><figcaption>T = real receipt (new detail)
</figcaption></figure><figure><img src='{sid}/C.png' style='max-width:480px'><figcaption>C = same box from the
already-shown source view (control)</figcaption></figure>
<details><summary>Old continuation and final answer (open AFTER labelling facts)</summary>{later_html}
<p>gold={t['gt']} old pred={html.escape(str(t.get('pred')))} old ok={ok}</p></details></section>""")
(out / "review" / "index.html").write_text("<html><meta charset='utf-8'><body style='font-family:sans-serif'>"
                                           "<h1>Paired-receipt MVE · human review (SFT-A)</h1>" + "".join(sections) +
                                           "</body></html>", encoding="utf-8")
json.dump(dict(design="20 first + 20 follow-up; 12 old-wrong + 8 old-right each; distinct uids; seed 20261022",
               counts={f"{k[0]}_{'right' if k[1] else 'wrong'}": v for k, v in got.items()}, states=states),
          open(out / "review" / "states.json", "w"), indent=1)
with open(out / "review" / "annotation_template.csv", "w") as f:
    f.write("id,new_fact_visible_in_T (text or none),fact_needed_for_answer (y/n/unclear),"
            "student_wrote_fact_correctly_after_T (y/n/unclear),later_misuse_of_fact (y/n/unclear),"
            "category (redundant|necessary|evidence_not_used|no_evidence_or_misleading|unknown),eligible_for_MVE (y/n),notes\n")
    for s in states:
        if s["rebuilt"]:
            f.write(f"{s['id']},,,,,,,\n")
print(json.dumps(dict(n=len(states), rebuilt=sum(s["rebuilt"] for s in states), counts={f"{k[0]}_{k[1]}": v for k, v in got.items()})))
