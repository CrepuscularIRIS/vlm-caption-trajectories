"""Per-step update ledger (10-28, user: prove which pool questions were trained on and that groups carry signal).

Driver-side wrapper of ray_trainer.compute_advantage (runs on the final batch of each step, after DAPO filtering):
appends one JSON line per step to CAPTION_UPDATE_LOG with
- qids: stable pool question ids (dataset `uid` column) of the groups entering the update;
- groups: all_correct / all_wrong / mixed counts over the step's groups (by correctness of each rollout);
- adv: share of response tokens with non-zero advantage, and mean |advantage| over response tokens.
Generation-time coverage (incl. DAPO-filtered groups) stays in CAPTION_ROLLOUT_LOG (field `qid`).
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

_STEP = 0


def step_record(qids, uids, correct, adv, mask) -> dict:
    """Pure summary of one update batch (CPU-testable)."""
    groups = defaultdict(list)
    for u, c in zip(uids, correct):
        groups[u].append(c)
    kinds = {"all_correct": 0, "all_wrong": 0, "mixed": 0}
    for v in groups.values():
        kinds["all_correct" if min(v) >= 1 else "all_wrong" if max(v) <= 0 else "mixed"] += 1
    m = mask.bool()
    n_tok = int(m.sum())
    nz = int(((adv != 0) & m).sum())
    return {"n_traj": len(uids), "n_groups": len(groups), "qids": sorted(set(qids)), "groups": kinds,
            "adv_nonzero_token_share": nz / max(1, n_tok), "adv_abs_mean": float(adv.abs()[m].mean()) if n_tok else 0.0}


def install() -> None:
    path = os.environ.get("CAPTION_UPDATE_LOG")
    if not path:
        return
    import verl.trainer.ray_trainer as RT
    if getattr(RT.compute_advantage, "_ledger", False):
        return
    original = RT.compute_advantage

    def compute_advantage(data, *args, **kwargs):
        global _STEP
        data = original(data, *args, **kwargs)
        mm = data.non_tensor_batch["multi_modal_data"]
        scores = data.batch["token_level_rewards"].sum(-1).tolist()
        rec = step_record([m.get("caption_qid", "") for m in mm], list(data.non_tensor_batch["uid"]),
                          [1.0 if s > 0 else 0.0 for s in scores], data.batch["advantages"], data.batch["response_mask"])
        _STEP += 1
        rec = {"call": _STEP, **rec}
        with open(path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"[ledger] call={_STEP} groups={rec['n_groups']} {rec['groups']} "
              f"adv_nonzero_tok={rec['adv_nonzero_token_share']:.3f} adv_abs={rec['adv_abs_mean']:.4f}", flush=True)
        return data

    compute_advantage._ledger = True
    RT.compute_advantage = compute_advantage


if __name__ == "__main__":
    import torch
    r = step_record(["a", "a", "b", "b"], ["u1", "u1", "u2", "u2"], [1.0, 0.0, 1.0, 1.0],
                    torch.tensor([[0.5, 0.5, 0.0], [-0.5, -0.5, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
                    torch.tensor([[1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0]]))
    assert r["groups"] == {"all_correct": 1, "all_wrong": 0, "mixed": 1} and r["qids"] == ["a", "b"]
    assert r["adv_nonzero_token_share"] == 0.5 and abs(r["adv_abs_mean"] - 0.25) < 1e-9
    print("update_ledger ok", r)
