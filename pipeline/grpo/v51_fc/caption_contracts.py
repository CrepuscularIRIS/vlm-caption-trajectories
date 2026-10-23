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
    """Terminal correctness x crop cost on generated tokens only, with a sparse loss mask."""
    from reward_caption import compute_score, correct_one
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
        inputs.append({"response": text, "ground_truth": data.non_tensor_batch["ground_truth"][i],
                       "crops": int(metadata[i].get("caption_crops", 0)) if metadata is not None else 0})
        last.append(int(where[-1]) if len(where) else None)
    scores = compute_score(inputs)
    corrects = []
    for item in inputs:
        gold = item["ground_truth"]  # may be {'style': 'rule', 'ground_truth': 'C'}
        gold = gold.get("ground_truth", "") if isinstance(gold, dict) else str(gold)
        corrects.append(correct_one("caption", item["response"], gold))
    _log_cropcost_rollouts(data, metadata, inputs, scores, corrects, mask)
    for i, (score, pos) in enumerate(zip(scores, last)):
        if pos is not None:
            reward[i, pos] = score
    return reward, {"overall": scores, "accuracy": corrects}


_train_reward_calls = 0


def _log_cropcost_rollouts(data, metadata, inputs, scores, corrects, mask):
    """Append one jsonl line per TRAIN sample: step, uid, i, terminal, crops, correct, score, gen_tokens.

    Train rollouts repeat each question's uuid uid n=16 times; validation batches keep
    unique pool uids, so any uid occurring more than once marks a training batch.
    """
    global _train_reward_calls
    import json
    import os
    from collections import Counter
    uids = [str(u) for u in data.non_tensor_batch["uid"]]
    if Counter(uids).most_common(1)[0][1] < 2:
        return  # validation chunk: not part of the training-step log
    _train_reward_calls += 1
    extra = data.non_tensor_batch.get("extra_info")
    stratum = data.non_tensor_batch.get("stratum")
    rows = []
    for i, uid in enumerate(uids):
        rows.append({"step": _train_reward_calls, "i": i, "uid": uid,
                     "terminal": metadata[i].get("caption_terminal") if metadata is not None else None,
                     "crops": inputs[i]["crops"], "correct": corrects[i], "score": scores[i],
                     "gen_tokens": int(mask[i].sum()),
                     "b": metadata[i].get("caption_b") if metadata is not None else None,
                     "stratum": str(stratum[i]) if stratum is not None else None,
                     "question": (extra[i].get("question") if extra is not None and isinstance(extra[i], dict) else None)})
    rollout_log = os.environ["CAPTION_ROLLOUT_LOG"]
    os.makedirs(os.path.dirname(rollout_log), exist_ok=True)
    with open(rollout_log, "a") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def actor_multimodal_inputs(worker, data):
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
