"""Observation-only training costs; never changes rewards, masks or sampling.

reward_call is session-local, NOT the trainer global step. Group uid is the
trainer's UUID, NOT the source question id. Main trainer logs/checkpoints remain
the authority for committed optimizer steps.
"""
import json
import os
import uuid
from collections import Counter

_session = os.environ.get("AUDIT_SESSION_ID") or uuid.uuid4().hex
_calls = 0


def log_batch(data, scores, metadata=None):
    global _calls
    path = os.environ.get("CAPTION_ROLLOUT_LOG")
    uids = [str(u) for u in data.non_tensor_batch["uid"]]
    # Current frozen recipe: train n=16, validation n=1. Do not log validation
    # under the training ledger. This predicate is not a general phase detector.
    if not path or not uids or Counter(uids).most_common(1)[0][1] < 2:
        return
    _calls += 1
    width = data.batch["prompts"].shape[-1]
    with open(path, "a") as stream:
        for i, uid in enumerate(uids):
            generated = int(data.batch["response_mask"][i].sum())
            initial = int(data.batch["attention_mask"][i, :width].sum())
            mm = metadata[i] if metadata is not None else None
            row = {
                "session_id": _session, "reward_call": _calls,
                "step_kind": "session_local_reward_call_not_global_step",
                "i": i, "group_uid": uid, "correct": float(scores[i]),
                "terminal": mm.get("caption_terminal") if mm is not None else "unknown",
                "crops": int(mm["caption_crops"]) if mm is not None else 0,
                "gen_tokens": generated,
                "proc_tokens": int(mm["caption_proc_tokens"]) if mm is not None else initial + generated,
            }
            stream.write(json.dumps(row) + "\n")


def install_direct_logger(manager):
    """Wrap stock reward computation and return the same objects unchanged."""
    if getattr(manager, "_direct_ledger_installed", False):
        return
    original = manager.compute_reward

    def compute_reward(self, data):
        result = original(self, data)
        reward_tensor, _metrics = result
        log_batch(data, reward_tensor.sum(dim=-1).tolist())
        return result

    manager.compute_reward = compute_reward
    manager._direct_ledger_installed = True
