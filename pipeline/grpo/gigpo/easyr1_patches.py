"""Engineering patches for grpo_v52m class-4 baselines (10-27), applied by easyr1_caption.install().

- GSPO (loss_avg_mode=seq): stock dp_actor scales each micro-batch loss by its response-token share of the
  mini-batch, which re-weights a per-sequence mean by length. With seq mode the share is the sequence count.
  token mode is unchanged.
- DAPO dynamic sampling: a rollout chunk whose groups are all filtered out raised RuntimeError; it is now skipped
  and the next chunk generated. trainer.max_try_make_batch still bounds the attempts.
"""
from __future__ import annotations

import inspect
import sys
import textwrap

SEQ_EDITS = [
    ("                dist.all_reduce(total_response_tokens, op=dist.ReduceOp.SUM)\n",
     "                dist.all_reduce(total_response_tokens, op=dist.ReduceOp.SUM)\n"
     "                total_sequences = torch.tensor(mini_batch.batch[\"response_mask\"].shape[0],\n"
     "                                               device=total_response_tokens.device,\n"
     "                                               dtype=total_response_tokens.dtype)\n"
     "                dist.all_reduce(total_sequences, op=dist.ReduceOp.SUM)\n"),
    ("                    loss = loss * torch.sum(response_mask) * self.world_size / total_response_tokens\n",
     "                    if self.config.loss_avg_mode == \"seq\":\n"
     "                        loss = loss * response_mask.shape[0] * self.world_size / total_sequences\n"
     "                    else:\n"
     "                        loss = loss * torch.sum(response_mask) * self.world_size / total_response_tokens\n"),
]

FILTER_EDITS = [
    ("                if len(kept_sample_idxs) == 0:\n"
     "                    raise RuntimeError(\"No sample is kept after filtering. Please check your data.\")\n",
     "                if len(kept_sample_idxs) == 0:\n"
     "                    print(\"[caption] filtering kept 0 groups in this chunk; generating the next chunk\", flush=True)\n"
     "                    continue\n"),
]


def patch_source(cls, name: str, edits: list[tuple[str, str]]) -> None:
    """Re-define cls.name from its installed source with exact one-occurrence text edits (fails loudly otherwise)."""
    fn = getattr(cls, name)
    if getattr(fn, "_caption_patched", False):
        return
    src = textwrap.dedent(inspect.getsource(fn))
    for old, new in edits:
        if src.count(old) != 1:
            raise RuntimeError(f"patch anchor for {cls.__name__}.{name} found {src.count(old)} times: {old[:70]!r}")
        src = src.replace(old, new)
    ns = dict(vars(sys.modules[fn.__module__]))
    exec(compile(src, f"<caption patch {cls.__name__}.{name}>", "exec"), ns)
    ns[name]._caption_patched = True
    setattr(cls, name, ns[name])


def install() -> None:
    from verl.trainer.ray_trainer import RayPPOTrainer
    from verl.workers.actor.dp_actor import DataParallelPPOActor
    patch_source(DataParallelPPOActor, "update_policy", [(_dedent4(o), _dedent4(n)) for o, n in SEQ_EDITS])
    patch_source(RayPPOTrainer, "_make_batch_data", [(_dedent4(o), _dedent4(n)) for o, n in FILTER_EDITS])


def _dedent4(text: str) -> str:
    """Anchors are written with class-body indentation; getsource+dedent removes one 4-space level."""
    return "\n".join(line[4:] if line.startswith("    ") else line for line in text.split("\n"))
