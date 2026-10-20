"""Fixed question list for the two-branch check: 300 questions from the RL training pool, stratified by source
(MME-RealWorld 200, WorldBench 50, MMMU-Pro 50), seed 20261019. Writes one uid per line."""
import sys
import pandas as pd

pool, out = sys.argv[1], sys.argv[2]
df = pd.read_parquet(f"{pool}/train.parquet")
quota = {"mme_realworld": 200, "worldbench": 50, "mmmu_pro": 50}
parts = [df[df.data_source == s].sample(n=k, random_state=20261019) for s, k in quota.items()]
sel = pd.concat(parts).sort_values("uid")
open(out, "w").write("\n".join(sel.uid) + "\n")
print(len(sel), sel.data_source.value_counts().to_dict(), sel.stratum.value_counts().to_dict())
