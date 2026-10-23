"""First-action decision loss (mentor 10-22; Pro design docs/research_design/decision_loss_20261022/PRO_RESPONSE.md).

At the first post-caption pre-action prefix h of ONE pre-selected rollout per question, the current policy is forked
K times with the forced action tag `<answer` and K times with `<grounding`, each continued by the policy to a terminal
state under protocol v5.2i. Per fork R = Y (1 - lambda N) (production reward, computed per fork BEFORE averaging);
D = stopgrad(mean R_answer - mean R_crop). The actor then adds, in the SAME optimizer step as RLOO,

    L = L_RLOO(token mean, unchanged) + beta / H0 * sum_q e_q * l_q,
    l = relu(D) softplus(-z) + relu(-D) softplus(z),   z = s_A - s_C,   s_a = sum_j log pi_T(t_a,j | h, t_a,<j)

with H0 = number of pre-assigned states (questions), unusable states contributing 0 but staying in H0.

Tag tokens (Qwen3-VL tokenizer, verified on the server 10-22): '<answer>' -> ['<', 'answer', '>'] but the policy
emits '<', 'answer', '>B' ('>' merges with the letter) and '<', 'ground', 'ing', '>{"' (merges with the JSON). The
complete distinguishing label is therefore ANSWER_TAG = '<answer' and CROP_TAG = '<grounding' (2 and 3 tokens; the
shared '<' cancels in z but enters the mass m). h ends with '</think>' '\n'; any other boundary is unusable.
CAPTION_DL_BETA = 0 is the protocol-matched control: no forks, no aux loss.
"""
from __future__ import annotations

import hashlib
import math
import os

import torch
import torch.nn.functional as F

ANSWER_TAG = "<answer"
CROP_TAG = "<grounding"
THINK_CLOSE = "</think>"


def beta() -> float:
    return float(os.environ.get("CAPTION_DL_BETA", "0"))


def fork_k() -> int:
    return int(os.environ.get("CAPTION_DL_K", "4"))


# ---------------------------------------------------------------- labels (pure, CPU-tested)

def branch_value(rewards: list[float]) -> float:
    """Q_hat = mean of per-fork R = Y (1 - lambda N); never mean(Y) * (1 - lambda mean(N))."""
    if not rewards:
        raise ValueError("empty branch")
    return sum(rewards) / len(rewards)


def decision_gap(r_answer: list[float], r_crop: list[float]) -> float:
    return branch_value(r_answer) - branch_value(r_crop)


def pair_loss(z: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """Stable |D| BCE(1[D>0], sigmoid(z)); d is a detached target. D == 0 gives exactly zero loss and gradient."""
    d = d.detach()
    return torch.relu(d) * F.softplus(-z) + torch.relu(-d) * F.softplus(z)


def pair_grad_z(z: float, d: float) -> float:
    """d l / d z = |D| (p - 1[D>0]) (closed form for tests and logs)."""
    p = 1.0 / (1.0 + math.exp(-z))
    return abs(d) * (p - (1.0 if d > 0 else 0.0))


def select_slot(seed: int, question_key: str, call_index: int, n: int) -> int:
    """Pre-assigned main-rollout slot per question, fixed before any sampling (namespace 'state')."""
    h = hashlib.sha256(f"{seed}|state|{call_index}|{question_key}".encode()).hexdigest()
    return int(h, 16) % n


def find_boundary(turn_ids: list[int], think_close_id: int, newline_id: int, lt_id: int) -> int | None:
    """Index j such that turn_ids[:j] = h (ends with '</think>' '\\n') and turn_ids[j] == '<' (the action tag start).
    None if the sampled ledger has any other boundary (unusable state, counted in the denominator)."""
    for j, tok in enumerate(turn_ids):
        if tok == think_close_id:
            if j + 2 < len(turn_ids) and turn_ids[j + 1] == newline_id and turn_ids[j + 2] == lt_id:
                return j + 2
            return None
    return None


# ---------------------------------------------------------------- actor patch

def _aux_microbatch(rec: dict, device) -> dict:
    """Two rows (answer tag, crop tag) sharing the prefix h; responses = tag tokens, right padded."""
    prefix, tags = rec["prefix"], (rec["tag_answer"], rec["tag_crop"])
    width = max(len(t) for t in tags)
    pad = rec["pad_id"]
    ids, att, resp, rmask, pos = [], [], [], [], []
    for t, positions in zip(tags, (rec["pos_answer"], rec["pos_crop"])):
        fill = width - len(t)
        ids.append(prefix + t + [pad] * fill)
        att.append([1] * (len(prefix) + len(t)) + [0] * fill)
        resp.append(t + [pad] * fill)
        rmask.append([1] * len(t) + [0] * fill)
        p = torch.zeros((positions.shape[0], len(prefix) + width), dtype=torch.long)
        p[:, :len(prefix) + len(t)] = positions
        pos.append(p)
    import numpy as np
    return {"input_ids": torch.tensor(ids, device=device), "attention_mask": torch.tensor(att, device=device),
            "responses": torch.tensor(resp, device=device), "position_ids": torch.stack(pos).to(device),
            "multi_modal_inputs": np.array([rec["mm"], rec["mm"]], dtype=object),
            "_rmask": torch.tensor(rmask, device=device, dtype=torch.float32)}


def _aux_backward(actor, real: list, dummy: dict | None, temperature: float, b: float, h0: int) -> dict:
    """Forward/backward every local aux state; ranks run the same number of passes (FSDP collectives), padding with
    zero-weight passes on the rank's dummy record."""
    import torch.distributed as dist
    device = torch.cuda.current_device()
    n_local = torch.tensor(len(real), device=device)
    dist.all_reduce(n_local, op=dist.ReduceOp.MAX)
    n_pass = int(n_local.item())
    ws = dist.get_world_size()
    stats = {"loss": 0.0, "p": 0.0, "mass": 0.0, "absd": 0.0, "n": 0, "agree": 0}
    for k in range(n_pass):
        rec = real[k] if k < len(real) else dummy
        if rec is None:
            raise RuntimeError("decision loss: rank has no record for a padding pass")
        mb = _aux_microbatch(rec, device)
        lp = actor._forward_micro_batch(mb, temperature=temperature)  # (2, width), temperature-scaled like RLOO
        s = lp.masked_fill(mb["_rmask"] == 0, 0.0).sum(dim=-1)
        z = s[0] - s[1]
        weight = 0.0 if k >= len(real) else b / h0 * ws  # x world_size: FSDP averages gradients over ranks
        d = torch.tensor(float(rec["d"]) if k < len(real) else 0.0, device=device)
        loss = pair_loss(z, d)
        (loss * weight).backward()
        if k < len(real):
            p = torch.sigmoid(z).item()
            stats["loss"] += loss.item()
            stats["p"] += p
            stats["mass"] += float(torch.exp(s[0]).item() + torch.exp(s[1]).item())
            stats["absd"] += abs(float(rec["d"]))
            stats["n"] += 1
            stats["agree"] += int((p > 0.5) == (float(rec["d"]) > 0)) if float(rec["d"]) != 0 else 0
    return stats


def install_actor_patch():
    """Stash aux records before update_policy; run the aux backward right before the single optimizer step."""
    from verl.workers.actor.dp_actor import DataParallelPPOActor as A
    if getattr(A, "_dl_patched", False):
        return
    original_update, original_step = A.update_policy, A._optimizer_step

    def update_policy(self, data):
        recs = data.non_tensor_batch.get("caption_aux")
        self._dl_pending = split_records(recs) if recs is not None and beta() > 0 else None
        self._dl_temperature = data.meta_info["temperature"]
        self._dl_steps = 0
        metrics = original_update(self, data)
        if self._dl_steps != 1 and self._dl_pending is not None:
            raise RuntimeError(f"decision loss expects exactly one optimizer step per update, got {self._dl_steps}")
        st = getattr(self, "_dl_stats", None)
        if st:
            k = max(1, st["n"])
            metrics["actor/dl_states"] = [st["n"]]
            metrics["actor/dl_loss"] = [st["loss"] / k]
            metrics["actor/dl_p_answer"] = [st["p"] / k]
            metrics["actor/dl_tag_mass"] = [st["mass"] / k]
            metrics["actor/dl_abs_d"] = [st["absd"] / k]
            print(f"[dl_aux] rank_states={st['n']} loss={st['loss'] / k:.5f} p_answer={st['p'] / k:.4f} "
                  f"mass={st['mass'] / k:.4f} abs_d={st['absd'] / k:.4f} agree={st['agree']}", flush=True)
        self._dl_pending, self._dl_stats = None, None
        return metrics

    def _optimizer_step(self):
        self._dl_steps = getattr(self, "_dl_steps", 0) + 1
        pending = getattr(self, "_dl_pending", None)
        if pending is not None:
            h0 = int(os.environ["CAPTION_DL_H0"])
            self._dl_stats = _aux_backward(self, pending[0], pending[1], self._dl_temperature, beta(), h0)
        return original_step(self)

    A.update_policy, A._optimizer_step, A._dl_patched = update_policy, _optimizer_step, True


def build_aux_records(worker, data) -> None:
    """FSDP worker side (has the processor): turn rollout aux states into actor inputs.

    Row entries are None or {"real": record | None, "dummy": record | None}; row 0 of every rank batch carries a dummy
    (its prompt plus its first two generated tokens, original image only) used for zero-weight padding passes."""
    import numpy as np
    mms = data.non_tensor_batch["multi_modal_data"]
    out = [None] * len(mms)
    axes = data.batch["position_ids"].shape[1]
    proc = worker.processor
    pad = proc.tokenizer.pad_token_id
    for i, mm in enumerate(mms):
        aux = mm.get("caption_dl")
        if aux and aux.get("d") is not None:
            out[i] = {"real": _record(proc, mm["images"][:1], aux["prefix"], aux["tag_answer"], aux["tag_crop"],
                                      aux["d"], axes, pad), "dummy": None}
    if len(mms):
        active = data.batch["attention_mask"][0].bool()
        ids = data.batch["input_ids"][0][active].tolist()
        cut = _prompt_length(data, 0)
        dummy = _record(proc, mms[0]["images"][:1], ids[:cut], ids[cut:cut + 2], ids[cut:cut + 2], None, axes, pad)
        out[0] = {"real": out[0]["real"] if out[0] else None, "dummy": dummy}
    data.non_tensor_batch["caption_aux"] = np.array(out, dtype=object)


def split_records(entries) -> tuple[list, dict | None]:
    real = [e["real"] for e in entries if e is not None and e["real"] is not None]
    dummy = next((e["dummy"] for e in entries if e is not None and e["dummy"] is not None), None)
    return real, dummy


def _prompt_length(data, i: int) -> int:
    """Number of active prompt tokens of row i (left-padded prompt part of input_ids)."""
    width = data.batch["prompts"].shape[-1]
    return int(data.batch["attention_mask"][i][:width].sum())


def _record(proc, images, prefix, tag_a, tag_c, d, axes, pad) -> dict:
    from caption_contracts import grid_list, positions_for_sequence
    features = dict(proc.image_processor(images=images, return_tensors="pt"))
    grid = grid_list(features["image_grid_thw"])
    pos_a = positions_for_sequence(proc, prefix + tag_a, grid, axes)
    pos_c = positions_for_sequence(proc, prefix + tag_c, grid, axes)
    return {"prefix": list(prefix), "tag_answer": list(tag_a), "tag_crop": list(tag_c), "d": d, "mm": features,
            "pos_answer": pos_a, "pos_crop": pos_c, "pad_id": pad}
