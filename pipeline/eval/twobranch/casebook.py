"""Text casebook for the mentor's hand inspection: for each uid in inspection_sample.csv, the question (from the RL
training pool prompt), gold letter, image size, and every sample of both branches: per turn the behaviour header,
caption, action (box in original-image fraction / answer), and correctness. Also an HTML page with the original image
and the crop boxes drawn, for a person to open on the server. Writes casebook.md and casebook.html next to the input."""
import base64, csv, html, io, json, re, sys
import pandas as pd
from PIL import Image, ImageDraw
d, pool = sys.argv[1], sys.argv[2]
samp = list(csv.DictReader(open(f"{d}/inspection_sample.csv")))
want = {r["uid"] for r in samp}
df = pd.read_parquet(f"{pool}/train.parquet").set_index("uid")
T = {b: {} for b in ("direct", "crop")}
for b in T:
    for l in open(f"{d}/traj_A_sft_{b}.jsonl"):
        r = json.loads(l)
        if r["uid"] in want: T[b].setdefault(r["uid"], []).append(r)
md, ht = [], ["<html><meta charset=utf-8><body style='font-family:sans-serif;max-width:1100px'>"]
for s in samp:
    u = s["uid"]; row = df.loc[u]
    q = re.sub(r"\s+", " ", str(row["env_head"]).split("Original image")[0]).strip()
    img = Image.open(row["images"][0]).convert("RGB"); W, H = img.size
    ql = [f"\n## [{s['category']}] {u} ({s['source']})  gold={row['reward_model']['ground_truth'] if isinstance(row['reward_model'], dict) else row['reward_model']}  acc_direct={float(s['acc_direct']):.2f} acc_crop={float(s['acc_crop']):.2f}  image {W}x{H}\n{q}\n"]
    boxes = []
    for b in ("direct", "crop"):
        for r in sorted(T[b].get(u, []), key=lambda x: x["sample_idx"]):
            tag = "ERR " + r["error"] if r.get("error") else ("OK" if r["ok"] else "wrong")
            ql.append(f"- {b}#{r['sample_idx']} {tag} pred={r['pred']} crops={r['n_crops']} rejected_first={r.get('first_tries',0)}")
            for t in r["turns"]:
                cap = re.sub(r"\s+", " ", t["text"])[:420]
                ql.append(f"    - {cap}")
            if b == "crop": boxes += [(r["sample_idx"], bx) for bx in r["boxes_abs"][1:]]
    md += ql
    im = img.copy(); dr = ImageDraw.Draw(im)
    for i, bx in boxes: dr.rectangle(bx, outline=["red", "lime", "blue", "orange"][i % 4], width=max(2, W // 300))
    im.thumbnail((900, 900)); buf = io.BytesIO(); im.save(buf, "JPEG", quality=80)
    ht.append(f"<h3>[{s['category']}] {html.escape(u)} — {html.escape(s['source'])}</h3><p>{html.escape(q)}</p>"
              f"<img src='data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}'><pre>"
              + html.escape("\n".join(x for x in md[-200:] if u in x or x.startswith("- ") or x.startswith("    -"))[:6000]) + "</pre>")
open(f"{d}/casebook.md", "w").write("\n".join(md)); ht.append("</body></html>"); open(f"{d}/casebook.html", "w").write("".join(ht))
print("written", len(samp), "questions")
