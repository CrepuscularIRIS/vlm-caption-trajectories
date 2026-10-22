"""Image review page for the mined 'existing-evidence utilization' candidates (zero GPU).
For each case in review_sample.json, replay the crop environment over the WHOLE trajectory and save every view the
student actually received (processed with process_image, min 1024, max 589,824 px): view_0 = initial view, view_k =
receipt of crop k. index.html shows per case: question with options, then each turn's full text next to the view it
produced, and (folded) gold, final answer and the text rule that flagged it, so the reviewer reads the evidence before
seeing the label. Also writes cases.json (no images) and annotation_template.csv.
Usage: python build_case_pages.py <mining_dir> <out_dir>"""
import html
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.environ["CAPTION_GRPO_CODE"])
sys.path.insert(0, os.environ["EASYR1_DIR"])
import pandas as pd  # noqa: E402
from verl.utils.dataset import process_image  # noqa: E402

import caption_env as CENV  # noqa: E402

MIN_PX, MAX_PX = 1024, 589824
POOL = os.environ["DATA_DIR"] + "/grpo_caption_v51_v3"  # [scrub]
mining, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
ROWS = {}
for split in ("train", "val", "vstar_holdout"):
    for _, r in pd.read_parquet(f"{POOL}/{split}.parquet").iterrows():
        ROWS[r["uid"]] = r
Q = json.load(open(mining / "questions.json"))
sample = json.load(open(mining / "review_sample.json"))
sections, cases = [], []
for n, c in enumerate(sample):
    cid = f"u{n:02d}"
    t = next(json.loads(l) for l in open(c["file"]) if json.loads(l)["uid"] == c["uid"]
             and str(json.loads(l)["sample_idx"]) == str(c["sample_idx"]))
    env = CENV.CropEnv(ROWS[c["uid"]]["images"][0])
    d = out / cid
    d.mkdir(exist_ok=True)
    process_image(env.images[0], MIN_PX, MAX_PX).save(d / "view_0.png")
    blocks = [f"<figure><img src='{cid}/view_0.png' style='max-width:520px'><figcaption>view 0 (initial, as shown)"
              f"</figcaption></figure>"]
    ok = True
    for k, turn in enumerate(t["turns"], start=1):
        p = turn.get("parsed") or {}
        blocks.append(f"<p><b>turn {k}</b> ({html.escape(str(p.get('kind')))})</p><pre>{html.escape(turn['text'])}</pre>")
        if p.get("kind") == "grounding" and "box_abs" in turn:
            r = env.step(dict(p))
            if r is None or list(env.boxes[-1]) != list(turn["box_abs"]):
                ok = False
                blocks.append("<p><i>replay mismatch: receipt not shown</i></p>")
                break
            vi = len(env.images) - 1
            process_image(env.images[-1], MIN_PX, MAX_PX).save(d / f"view_{vi}.png")
            blocks.append(f"<figure><img src='{cid}/view_{vi}.png' style='max-width:520px'><figcaption>view {vi} = "
                          f"receipt of crop {vi} (as shown)</figcaption></figure>")
    q = Q[c["uid"]]
    cases.append(dict(id=cid, set=c["set"], rule=c["rule"], uid=c["uid"], sample_idx=c["sample_idx"], source=c["source"],
                      gold=c["gold"], pred=c["pred"], n_crops=c["n_crops"], replay_ok=ok, n_views=len(env.images)))
    sections.append(f"""<section id='{cid}'><h2>{cid} · {c['set']} · {c['source']}</h2>
<pre>{html.escape(q['question'])}</pre>{''.join(blocks)}
<details><summary>gold / final answer / flagging rule (open AFTER reading the evidence)</summary>
<p>gold={c['gold']} final={c['pred']} rule={c['rule']}</p></details></section>""")
(out / "index.html").write_text("<html><meta charset='utf-8'><body style='font-family:sans-serif'><h1>Existing-evidence "
                                "utilization candidates</h1>" + "".join(sections) + "</body></html>", encoding="utf-8")
json.dump(cases, open(out / "cases.json", "w"), indent=1)
with open(out / "annotation_template.csv", "w") as f:
    f.write("id,gold_fact_correctly_stated_in_caption (y/n/unclear),statement_is_assertion_not_rejection (y/n/unclear),"
            "failure_type (option_mapping|row_column_binding|goal_drift|misread_existing_evidence|hallucinated_receipt|"
            "rule_false_positive|other),turn_where_correct_fact_first_stated,turn_where_error_enters,"
            "eligible_for_segment_value_MVE (y/n),notes\n")
    for x in cases:
        f.write(f"{x['id']},,,,,,,\n")
print(json.dumps(dict(n=len(cases), replay_ok=sum(x["replay_ok"] for x in cases))))
