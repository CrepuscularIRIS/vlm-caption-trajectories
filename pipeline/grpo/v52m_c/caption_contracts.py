"""Token/image contracts shared by rollout, CPU preflight and actor.

Environment presentation and protocol strings are deliberately untouched.
Model alignment is read from the processor/config, never inferred from 14/28.
"""
from __future__ import annotations

import copy
from typing import Any

import torch


def vision_contract(processor, model_config=None) -> dict:
    ip = processor.image_processor
    patch, merge = int(ip.patch_size), int(ip.merge_size)
    temporal = int(ip.temporal_patch_size)
    if min(patch, merge, temporal) <= 0:
        raise ValueError("Invalid vision processor dimensions")
    if model_config is not None:
        vc = model_config.vision_config
        actual = (int(vc.patch_size), int(vc.spatial_merge_size), int(vc.temporal_patch_size))
        if actual != (patch, merge, temporal):
            raise ValueError(f"Model/processor contract mismatch: model={actual}, processor={(patch, merge, temporal)}")
        if int(model_config.image_token_id) != int(processor.image_token_id):
            raise ValueError("Model/processor image token id mismatch")
    return {"patch_size": patch, "merge_size": merge, "temporal_patch_size": temporal,
            "model_alignment": patch * merge, "processor_class": type(ip).__name__}


def grid_list(grid) -> list[list[int]]:
    if isinstance(grid, torch.Tensor):
        grid = grid.detach().cpu().tolist()
    return [[int(v) for v in row] for row in grid]


def image_spans(ids: list[int], processor) -> list[tuple[int, int]]:
    """Half-open image-pad spans; reject malformed or stray vision tokens."""
    start = int(processor.vision_start_token_id)
    pad = int(processor.image_token_id)
    end = int(processor.tokenizer.convert_tokens_to_ids("<|vision_end|>"))
    spans, i = [], 0
    while i < len(ids):
        if ids[i] == pad or ids[i] == end:
            raise ValueError(f"Stray vision token at {i}")
        if ids[i] != start:
            i += 1
            continue
        j = i + 1
        while j < len(ids) and ids[j] == pad:
            j += 1
        if j == i + 1 or j >= len(ids) or ids[j] != end:
            raise ValueError(f"Incomplete image span at {i}")
        spans.append((i + 1, j))
        i = j + 1
    return spans


def validate_images(ids: list[int], grid, processor) -> list[tuple[int, int]]:
    rows = grid_list(grid)
    spans = image_spans(ids, processor)
    merge = vision_contract(processor)["merge_size"]
    if len(spans) != len(rows):
        raise ValueError(f"Image count mismatch: spans={len(spans)}, grids={len(rows)}")
    for k, ((a, b), row) in enumerate(zip(spans, rows)):
        if len(row) != 3:
            raise ValueError(f"Invalid grid shape at image {k}: {row}")
        t, h, w = row
        if min(row) <= 0 or h % merge or w % merge:
            raise ValueError(f"Invalid merged grid at image {k}: {row}")
        expected = t * h * w // (merge * merge)
        if b - a != expected:
            raise ValueError(f"Image {k}: pad_tokens={b-a}, expected={expected}, grid={row}")
    return spans


def raw_image_prompt(ids: list[int], grid, processor) -> list[int]:
    """Collapse HF-expanded spans to ONE marker/image for vLLM expansion.

    Every non-image token (including sampled assistant ids) stays byte-for-byte
    identical. Never decode/re-encode sampled assistant text to build history.
    """
    spans = validate_images(ids, grid, processor)
    result, pos = [], 0
    for a, b in spans:
        result.extend(ids[pos:a])
        result.append(int(processor.image_token_id))
        pos = b
    return result + ids[pos:]


def vllm_preflight(mm_processor, ids, images, grid, processor) -> dict:
    """vLLM 0.11 CPU processing, before scheduling any CUDA forward."""
    raw = raw_image_prompt(ids, grid, processor)
    mm = mm_processor.apply(raw, {"image": images}, {})
    if list(mm["prompt_token_ids"]) != list(ids):
        raise ValueError("HF/vLLM expanded prompt ids differ; request NOT submitted to GPU")
    vgrid = [grid_list(x["image_grid_thw"].data.reshape(1, 3))[0]
             for x in mm["mm_kwargs"]["image"]]
    if vgrid != grid_list(grid):
        raise ValueError(f"HF/vLLM grids differ: hf={grid_list(grid)}, vllm={vgrid}")
    validate_images(list(mm["prompt_token_ids"]), vgrid, processor)
    return {"prompt_token_ids": raw, "multi_modal_data": {"image": images}}


def sampling_for_group(base, metadata: dict) -> tuple[int, Any]:
    sampling = copy.deepcopy(base)
    for name, value in metadata.items():
        if hasattr(sampling, name):
            setattr(sampling, name, value)
    n = int(sampling.n)
    if n < 1:
        raise ValueError("GRPO group size must be positive")
    sampling.n = 1  # group expansion belongs to the outer trajectory loop
    return n, sampling


def flatten_segments(segments, limit: int) -> tuple[list[int], list[int]]:
    ids, mask = [], []
    for tokens, is_assistant in segments:
        ids.extend(tokens)
        mask.extend([int(is_assistant)] * len(tokens))
    if len(ids) > limit:
        # Never cut through an observation image or silently keep orphan images.
        raise ValueError(f"Trajectory exceeded reserved response budget: {len(ids)} > {limit}")
    return ids, mask


def positions_for_sequence(processor, ids, grid, axes: int, rope_fn=None):
    if rope_fn is None:
        from verl.models.transformers.qwen3_vl import get_rope_index
        rope_fn = get_rope_index
    seq = torch.as_tensor(ids, dtype=torch.long, device="cpu")
    validate_images(seq.tolist(), grid, processor)
    vision = rope_fn(processor, input_ids=seq,
                     image_grid_thw=torch.as_tensor(grid, dtype=torch.long),
                     attention_mask=torch.ones_like(seq))
    if tuple(vision.shape) != (3, len(seq)):
        raise ValueError(f"Unexpected mRoPE shape: {tuple(vision.shape)}")
    if axes == 3:
        return vision
    if axes == 4:  # EasyR1's padding-free interface has a separate text row
        return torch.cat((torch.arange(len(seq)).unsqueeze(0), vision), dim=0)
    raise ValueError(f"Qwen3-VL requires 3 or 4 position axes, got {axes}")


def assistant_reward_batch(manager, data):
    """Terminal correctness on generated tokens only, with a sparse loss mask."""
    from reward_caption import compute_score
    responses = data.batch["responses"]
    mask = data.batch["response_mask"].bool()
    reward = torch.zeros_like(responses, dtype=torch.float32)
    inputs, last = [], []
    metadata = data.non_tensor_batch.get("multi_modal_data")
    for i in range(responses.shape[0]):
        where = torch.nonzero(mask[i], as_tuple=False).flatten()
        text = manager.tokenizer.decode(responses[i][mask[i]].tolist(), skip_special_tokens=True)
        if metadata is not None and metadata[i].get("caption_terminal") != "answer":
            text = ""  # no valid terminal answer was reached
        inputs.append({"response": text, "ground_truth": data.non_tensor_batch["ground_truth"][i]})
        last.append(int(where[-1]) if len(where) else None)
    scores = compute_score(inputs)
    for i, (score, pos) in enumerate(zip(scores, last)):
        if pos is not None:
            reward[i, pos] = score
    _log_rollouts(data, metadata, scores, mask)
    return reward, {"overall": scores, "accuracy": scores}


_train_reward_calls = 0


def _log_rollouts(data, metadata, scores, mask):
    """One jsonl line per TRAIN trajectory (CAPTION_ROLLOUT_LOG): cost ledger for the trade-off figure.

    gen_tokens = generated (assistant) tokens; proc_tokens = sum over turns of full input + output.
    Train batches repeat each uid n times; validation uids are unique, so they are skipped.
    """
    global _train_reward_calls
    import json
    import os
    from collections import Counter
    path = os.environ.get("CAPTION_ROLLOUT_LOG")
    uids = [str(u) for u in data.non_tensor_batch["uid"]]
    if not path or metadata is None or Counter(uids).most_common(1)[0][1] < 2:
        return
    _train_reward_calls += 1
    with open(path, "a") as f:
        for i, uid in enumerate(uids):
            f.write(json.dumps({"step": _train_reward_calls, "i": i, "uid": uid,
                                "terminal": metadata[i].get("caption_terminal"),
                                "qid": metadata[i].get("caption_qid", ""),
                                "crops": int(metadata[i].get("caption_crops", 0)), "correct": float(scores[i]),
                                "gen_tokens": int(mask[i].sum()),
                                "proc_tokens": int(metadata[i].get("caption_proc_tokens", 0))}) + "\n")


def lazy_mm() -> bool:
    """CAPTION_LAZY_MM=1 (memory, 10-27): rollouts return view recipes instead of PIL images; the actor renders and
    preprocesses one micro-batch at a time. Same pixels, grids and mRoPE checks; 0 = grpo_v52 behaviour."""
    import os
    return os.environ.get("CAPTION_LAZY_MM", "0") == "1"


def view_recipe(path, executed_boxes, presented_count, min_px, max_px):
    """Reconstruct only views admitted to the token ledger, not every executed crop.

    A crop may execute (and count toward cost) before its receipt is rejected
    for context length. That crop is NOT an actor input.
    """
    if not 1 <= presented_count <= len(executed_boxes) + 1:
        raise ValueError("Invalid presented view count")
    return {"path": path, "boxes": [list(b) for b in executed_boxes[:presented_count - 1]],
            "min_px": min_px, "max_px": max_px}


_VIEW_CACHE: "OrderedDict" = None
_VIEW_BYTES = 0
import threading as _threading
_VIEW_LOCK = _threading.Lock()  # prefetch threads share the cache (10-28 fix: unlocked popitem raised KeyError)


def _cache_budget() -> int:
    """CAPTION_MM_CACHE_GB (default 0 = off): per-process LRU budget for per-view preprocessed pixel_values."""
    import os
    return int(float(os.environ.get("CAPTION_MM_CACHE_GB", "0")) * 2**30)


def _view_features(processor, path, box, min_px, max_px):
    """(pixel_values, image_grid_thw) of one view. Deterministic preprocessing of fixed pixels, so a cached entry is
    the same tensor the processor would return; vision-encoder outputs are never cached."""
    global _VIEW_CACHE, _VIEW_BYTES
    from collections import OrderedDict
    from caption_env import render_view
    budget = _cache_budget()
    key = (path, None if box is None else tuple(int(v) for v in box), min_px, max_px)
    if budget > 0:
        with _VIEW_LOCK:
            if _VIEW_CACHE is None:
                _VIEW_CACHE = OrderedDict()
            hit = _VIEW_CACHE.get(key)
            if hit is not None:
                _VIEW_CACHE.move_to_end(key)
                return hit
    f = processor.image_processor(images=[render_view(path, box, min_px, max_px)], return_tensors="pt")
    val = (f["pixel_values"], f["image_grid_thw"])
    if budget > 0:
        size = val[0].numel() * val[0].element_size()
        with _VIEW_LOCK:
            if key not in _VIEW_CACHE:
                _VIEW_CACHE[key] = val
                _VIEW_BYTES += size
            while _VIEW_BYTES > budget and len(_VIEW_CACHE) > 1:
                _, old = _VIEW_CACHE.popitem(last=False)
                _VIEW_BYTES -= old[0].numel() * old[0].element_size()
    return val


def recipe_features(processor, mm: dict) -> dict:
    r = mm["caption_recipe"]
    if _cache_budget() > 0:
        parts = [_view_features(processor, r["path"], b, r["min_px"], r["max_px"]) for b in [None] + list(r["boxes"])]
        features = {"pixel_values": torch.cat([p[0] for p in parts]), "image_grid_thw": torch.cat([p[1] for p in parts])}
    else:
        from caption_env import render_views
        images = render_views(r["path"], r["boxes"], r["min_px"], r["max_px"])
        features = dict(processor.image_processor(images=images, return_tensors="pt"))
    if grid_list(features["image_grid_thw"]) != mm["caption_grid"]:
        raise ValueError("Re-rendered views differ from the rollout grid ledger")
    return features


def install_lazy_forward():
    """Materialise multi_modal_inputs recipes inside each micro-batch forward (actor log-prob and update passes)."""
    from verl.workers.actor.dp_actor import DataParallelPPOActor as A
    if getattr(A, "_lazy_mm_patched", False):
        return
    original = A._forward_micro_batch

    def forward(self, micro_batch, temperature):
        mmi = micro_batch.get("multi_modal_inputs")
        if mmi is not None and len(mmi) and isinstance(mmi[0], dict) and "_recipe" in mmi[0]:
            import numpy as np
            micro_batch = dict(micro_batch)
            micro_batch["multi_modal_inputs"] = np.array(
                [recipe_features(self._lazy_processor, m["_recipe"]) for m in mmi], dtype=object)
        return original(self, micro_batch, temperature)

    A._forward_micro_batch, A._lazy_mm_patched = forward, True


def actor_multimodal_inputs(worker, data):
    if lazy_mm():
        return _actor_recipes(worker, data)
    return _actor_features(worker, data)


def _actor_recipes(worker, data):
    """Lazy variant: keep only recipes; mRoPE positions are checked here from the ledger grids (no pixels)."""
    import numpy as np
    for actor_attr in ("actor", "ref_policy"):
        actor = getattr(worker, actor_attr, None)
        if actor is not None:
            actor._lazy_processor = worker.processor
    result = []
    for i, mm in enumerate(data.non_tensor_batch["multi_modal_data"]):
        grids = mm["caption_grid"]
        active = data.batch["attention_mask"][i].bool()
        ids = data.batch["input_ids"][i][active].detach().cpu().tolist()
        validate_images(ids, grids, worker.processor)
        axes = data.batch["position_ids"].shape[1]
        expected = positions_for_sequence(worker.processor, ids, grids, axes)
        actual = data.batch["position_ids"][i][:, active].detach().cpu()
        if not torch.equal(actual, expected):
            raise ValueError(f"Actor/rollout mRoPE mismatch for trajectory {i}")
        result.append({"_recipe": mm})
    data.non_tensor_batch["multi_modal_inputs"] = np.array(result, dtype=object)


def _actor_features(worker, data):
    """Do not cache by question uid: siblings in a GRPO group have different crops."""
    import numpy as np
    result = []
    for i, mm in enumerate(data.non_tensor_batch["multi_modal_data"]):
        # These are the exact PIL inputs submitted during rollout. NO second fit().
        features = dict(worker.processor.image_processor(images=mm["images"], return_tensors="pt"))
        grids = grid_list(features["image_grid_thw"])
        if grids != mm["caption_grid"]:
            raise ValueError(f"Actor/rollout grid mismatch for trajectory {i}")
        active = data.batch["attention_mask"][i].bool()
        ids = data.batch["input_ids"][i][active].detach().cpu().tolist()
        validate_images(ids, grids, worker.processor)
        axes = data.batch["position_ids"].shape[1]
        expected = positions_for_sequence(worker.processor, ids, grids, axes)
        actual = data.batch["position_ids"][i][:, active].detach().cpu()
        if not torch.equal(actual, expected):
            raise ValueError(f"Actor/rollout mRoPE mismatch for trajectory {i}")
        result.append(features)
    data.non_tensor_batch["multi_modal_inputs"] = np.array(result, dtype=object)


class _Prefetcher:
    """CAPTION_MM_PREFETCH=N (threads; 0 = off): while the GPU runs micro-batch k, worker threads build the features of
    the next rows in data order. At most `window` finished-but-unconsumed rows are held (bounded memory). A row the
    forward needs before it was prefetched is computed synchronously, so ordering never blocks the forward."""

    def __init__(self, processor, rows, threads: int, window: int):
        import threading
        from concurrent.futures import ThreadPoolExecutor
        self.proc, self.pool = processor, ThreadPoolExecutor(threads)
        self.slots, self.futures, self.lock = threading.Semaphore(window), {}, threading.Lock()
        self.stop = False
        self.thread = threading.Thread(target=self._produce, args=(list(rows),), daemon=True)
        self.thread.start()

    def _produce(self, rows):
        for m in rows:
            self.slots.acquire()
            if self.stop:
                return
            with self.lock:
                self.futures[id(m)] = self.pool.submit(recipe_features, self.proc, m["_recipe"])

    def get(self, m):
        with self.lock:
            fut = self.futures.pop(id(m), None)
        if fut is None:
            return recipe_features(self.proc, m["_recipe"])
        out = fut.result()
        self.slots.release()
        return out

    def close(self):
        self.stop = True
        self.slots.release()
        self.pool.shutdown(wait=False, cancel_futures=True)


def install_prefetch():
    """Wrap actor compute_log_prob / update_policy so each pass prefetches its rows' features (lazy mode only)."""
    import os
    from verl.workers.actor.dp_actor import DataParallelPPOActor as A
    threads = int(os.environ.get("CAPTION_MM_PREFETCH", "0"))
    if threads <= 0 or getattr(A, "_prefetch_patched", False):
        return
    window = int(os.environ.get("CAPTION_MM_PREFETCH_WINDOW", "32"))
    original_forward = A._forward_micro_batch

    def wrap(name):
        original = getattr(A, name)

        def run(self, data, *args, **kwargs):
            rows = data.non_tensor_batch.get("multi_modal_inputs")
            pf = None
            if rows is not None and len(rows) and isinstance(rows[0], dict) and "_recipe" in rows[0]:
                pf = _Prefetcher(self._lazy_processor, rows, threads, window)
            self._prefetcher = pf
            try:
                return original(self, data, *args, **kwargs)
            finally:
                if pf is not None:
                    pf.close()
                self._prefetcher = None
        setattr(A, name, run)

    def forward(self, micro_batch, temperature):
        pf = getattr(self, "_prefetcher", None)
        mmi = micro_batch.get("multi_modal_inputs")
        if pf is not None and mmi is not None and len(mmi) and isinstance(mmi[0], dict) and "_recipe" in mmi[0]:
            import numpy as np
            micro_batch = dict(micro_batch)
            micro_batch["multi_modal_inputs"] = np.array([pf.get(m) for m in mmi], dtype=object)
        return original_forward(self, micro_batch, temperature)

    wrap("compute_log_prob")
    wrap("update_policy")
    A._forward_micro_batch, A._prefetch_patched = forward, True
