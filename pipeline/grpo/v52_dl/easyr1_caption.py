"""EasyR1 Qwen3-VL caption GRPO adapter, protocol v5.2i (caption_v52i.py: Step-1 WORD is intent only), plus the
first-action decision loss (decision_loss.py; mentor 10-22, Pro design 10-22). CAPTION_DL_BETA=0 is the protocol-matched
RLOO control (no forks, no aux loss).

Run from the installed EasyR1 checkout with this repository on PYTHONPATH.
Requires the existing Ray-worker hook calling install() in every worker.
GPU integration remains a release gate; see REVIEW_20261012.md.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path

import numpy as np
import torch
import caption_env as CENV
import hashlib
import json

import caption_v52i as proto
import decision_loss as DL
from caption_contracts import (
    actor_multimodal_inputs, assistant_reward_batch, flatten_segments, grid_list,
    positions_for_sequence, raw_image_prompt, sampling_for_group, validate_images,
    vision_contract, vllm_preflight,
)

TURN_MAX_TOKENS = 640


def _engine_overrides() -> dict:
    """CAPTION_ENGINE_KWARGS="k=v,k=v"; entries win over verl's engine kwargs."""
    raw = os.environ.get("CAPTION_ENGINE_KWARGS", "").strip()
    if not raw:
        return {}
    overrides = {}
    for item in raw.split(","):
        key, _, value = item.partition("=")
        key, value = key.strip(), value.strip()
        if value.lower() in ("true", "false"):
            overrides[key] = value.lower() == "true"
        else:
            try:
                overrides[key] = int(value)
            except ValueError:
                try:
                    overrides[key] = float(value)
                except ValueError:
                    overrides[key] = value
    return overrides


def encode_initial(proc, env, head, tail, min_px, max_px):
    from verl.utils.dataset import process_image
    messages = [{"role": "system", "content": proto.SYSTEM},
                {"role": "user", "content": [{"type": "text", "text": head + "\n"},
                 {"type": "image"}, {"type": "text", "text": tail}]}]
    text = proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    images = [process_image(env.images[0], min_px, max_px)]
    encoded = proc(images=images, text=[text], add_special_tokens=False, return_tensors="pt")
    ids = encoded["input_ids"][0].tolist()
    grid = grid_list(encoded["image_grid_thw"])
    validate_images(ids, grid, proc)
    return ids, images, grid


def receipt_chunk(proc, receipt, last_token):
    # A missing EOS is scaffolding, never a fabricated sampled token.
    im_end = proc.tokenizer.convert_tokens_to_ids("<|im_end|>")
    close = "" if last_token == im_end else "<|im_end|>"
    return (close + "\n<|im_start|>user\n" + receipt["head"] +
            "\n<|vision_start|><|image_pad|><|vision_end|>" + receipt["tail"] +
            "<|im_end|>\n<|im_start|>assistant\n")


def _fork_seed(base: int, *parts) -> int:
    """Fork RNG namespace, disjoint from the main rollout's seed counter (the main stream is never advanced)."""
    h = hashlib.sha256("|".join(str(p) for p in (base, "fork") + parts).encode()).hexdigest()
    return int(h, 16) % (2**31)


def _question_key(mm: dict) -> str:
    return hashlib.sha256((str(mm["images"][0]) + "\n" + mm["env"]["head"]).encode()).hexdigest()[:16]


def _fork_reward(proc, fork: dict, gold: str) -> tuple[int, float]:
    """Production reward on the fork's assistant tokens: Y = one <answer> letter == gold after a legal terminal
    answer; R = Y (1 - lambda N) with N = executed crops (reward_caption.compute_score)."""
    from reward_caption import compute_score, correct_one
    text = ""
    if fork["terminal"] == "answer":
        flat = [t for turn in fork["assistant"] for t in turn]
        text = proc.tokenizer.decode(flat, skip_special_tokens=True)
    y = int(correct_one("caption", text, gold))
    r = compute_score([{"response": text, "ground_truth": gold, "crops": int(fork["env"].crops)}])[0]
    return y, float(r)


def _advance_forks(self, forks, round_index, proc, base_sampling, min_px, max_px):
    """One lock-step round of the v5.2i environment for fork states (same rules as the main loop; round 0 continues
    turn 1 after the forced tag, inheriting the turn's remaining 640-token budget)."""
    from verl.utils.dataset import process_image
    requests, samplings, scheduled = [], [], []
    for f in forks:
        if f["terminal"] is not None:
            continue
        remaining = f["cap"] - len(f["ids"])
        budget = min(f["turn_budget"] if round_index == 0 else TURN_MAX_TOKENS, remaining)
        if remaining <= 0:
            f["terminal"] = "context_limit"
            continue
        if budget <= 0:
            f["terminal"] = "length"
            continue
        requests.append(vllm_preflight(self._caption_mm_processor, f["ids"], f["images"], f["grid"], proc))
        sampling = copy.deepcopy(base_sampling)
        sampling.seed = (f["seed"] + round_index) % (2**31)
        sampling.max_tokens = budget
        samplings.append(sampling)
        scheduled.append(f)
    if not scheduled:
        return 0
    outputs = self.inference_engine.generate(prompts=requests, sampling_params=samplings, use_tqdm=False)
    if len(outputs) != len(scheduled):
        raise RuntimeError("vLLM fork output count differs from requests")
    for f, output in zip(scheduled, outputs):
        if list(output.prompt_token_ids) != f["ids"]:
            raise RuntimeError("vLLM fork prompt differs from the token ledger")
        completion = output.outputs[0]
        sampled = list(completion.token_ids)
        if len(f["ids"]) + len(sampled) > f["cap"]:
            raise RuntimeError("Engine exceeded the reserved fork context budget")
        f["ids"] += sampled
        f["gen_tokens"] += len(sampled)
        turn = (f["turn_prefix"] if round_index == 0 else []) + sampled
        f["assistant"].append(turn)
        if not sampled or completion.finish_reason == "length":
            f["terminal"] = "length"
            continue
        text = proc.tokenizer.decode(turn, skip_special_tokens=True).strip()
        try:
            parsed = proto.parse_step(text)
            errors = proto.check_step(parsed, expected_step=round_index + 1, crops_used=f["env"].crops,
                                      available=f["env"].available, budget=CENV.MAX_CROPS)
        except (ValueError, KeyError, TypeError):
            parsed, errors = {}, ["parse_error"]
        if round_index == 0 and parsed.get("kind") != f["action"]:
            errors = list(errors) + ["forced_tag_not_followed"]
        if errors:
            f["terminal"] = "invalid"
            continue
        if parsed.get("kind") == "answer":
            f["terminal"] = "answer"
            continue
        if parsed.get("kind") != "grounding":
            f["terminal"] = "invalid"
            continue
        receipt = f["env"].step(parsed)
        if receipt is None:
            f["terminal"] = "invalid"
            continue
        image = process_image(receipt["image"], min_px, max_px)
        encoded = proc(images=[image], text=[receipt_chunk(proc, receipt, sampled[-1])],
                       add_special_tokens=False, return_tensors="pt")
        receipt_ids = encoded["input_ids"][0].tolist()
        receipt_grid = grid_list(encoded["image_grid_thw"])
        validate_images(receipt_ids, receipt_grid, proc)
        if len(f["ids"]) + len(receipt_ids) + 1 > f["cap"]:
            f["terminal"] = "context_limit"
            continue
        f["ids"] += receipt_ids
        f["images"].append(image)
        f["grid"] += receipt_grid
    return len(scheduled)


def run_dl_forks(self, states, source_mm, proc, base_sampling, n, min_px, max_px):
    """Decision-loss labels: at the first post-caption prefix h of ONE pre-assigned rollout per question, K forks with
    '<answer' and K with '<grounding', continued by the current policy to a terminal state; D = mean R_A - mean R_C.
    The pre-assigned slot is fixed before sampling; unusable slots stay unusable (no re-pick), counted in H0."""
    import time
    t0 = time.time()
    tok = proc.tokenizer
    think_close = tok.convert_tokens_to_ids(DL.THINK_CLOSE)
    newline = tok.encode("\n", add_special_tokens=False)
    tag_a = tok.encode(DL.ANSWER_TAG, add_special_tokens=False)
    tag_c = tok.encode(DL.CROP_TAG, add_special_tokens=False)
    if len(newline) != 1 or tag_a[0] != tag_c[0] or tag_a == tag_c:
        raise RuntimeError(f"unexpected tag tokenization newline={newline} answer={tag_a} crop={tag_c}")
    call = getattr(run_dl_forks, "_calls", 0)
    run_dl_forks._calls = call + 1
    seed, k = int(self.config.seed or 42), DL.fork_k()
    forks, status, keys = [], {}, {}
    for q in range(len(states) // n):
        key = _question_key(source_mm[q * n])
        slot = DL.select_slot(seed, key, call, n)
        i = q * n + slot
        keys[i] = key
        st = states[i]
        turns = [seg for seg, is_assistant in st["segments"] if is_assistant]
        j = DL.find_boundary(turns[0], think_close, newline[0], tag_a[0]) if turns else None
        status[i] = "boundary" if j is None else "ok"
        if j is None:
            continue
        prefix = st["ids"][:st["initial_length"]] + turns[0][:j]
        n_img = 1
        st["dl_aux"] = {"prefix": prefix, "tag_answer": tag_a, "tag_crop": tag_c, "d": None, "slot": slot}
        for action, tag in (("answer", tag_a), ("grounding", tag_c)):
            for r in range(k):
                forks.append({"owner": i, "action": action, "k": r, "ids": prefix + tag,
                              "images": st["images"][:n_img], "grid": st["grid"][:n_img],
                              "env": CENV.CropEnv(source_mm[i]["images"][0]), "turn_prefix": turns[0][:j] + tag,
                              "turn_budget": TURN_MAX_TOKENS - j - len(tag), "cap": st["cap"], "terminal": None,
                              "assistant": [], "gen_tokens": 0, "key": key,
                              "seed": _fork_seed(seed, call, key, action, r)})
    rounds = 0
    for round_index in range(CENV.MAX_CROPS + 1):
        if not _advance_forks(self, forks, round_index, proc, base_sampling, min_px, max_px):
            break
        rounds += 1
    for f in forks:
        f["terminal"] = f["terminal"] or "no_answer"
        f["y"], f["r"] = _fork_reward(proc, f, source_mm[f["owner"]]["env"]["gold"])
    by_owner = {}
    for f in forks:
        by_owner.setdefault(f["owner"], {"answer": [], "grounding": []})[f["action"]].append(f["r"])
    signs = {"pos": 0, "neg": 0, "zero": 0}
    for i, rs in by_owner.items():
        d = DL.decision_gap(rs["answer"], rs["grounding"])
        states[i]["dl_aux"]["d"] = d
        signs["pos" if d > 0 else "neg" if d < 0 else "zero"] += 1
    pid = os.getpid()
    lines = [json.dumps({"call": call, "pid": pid, "key": f["key"], "owner_row": f["owner"], "action": f["action"],
                         "k": f["k"], "terminal": f["terminal"], "y": f["y"], "n_crops": f["env"].crops, "r": f["r"],
                         "gen_tokens": f["gen_tokens"], "d": states[f["owner"]]["dl_aux"]["d"]}) for f in forks]
    lines += [json.dumps({"call": call, "pid": pid, "key": keys[i], "owner_row": i, "state_status": s})
              for i, s in status.items()]
    with open(f"{os.environ['CAPTION_DL_LOG']}.{pid}", "a") as fh:  # one file per rollout process, one write per call
        fh.write("\n".join(lines) + "\n")
    qa = [sum(v["answer"]) / len(v["answer"]) for v in by_owner.values()]
    qc = [sum(v["grounding"]) / len(v["grounding"]) for v in by_owner.values()]
    print(f"[dl_forks] call={call} states={len(status)} usable={sum(s == 'ok' for s in status.values())} "
          f"forks={len(forks)} rounds={rounds} gen_tokens={sum(f['gen_tokens'] for f in forks)} "
          f"q_answer={sum(qa) / max(1, len(qa)):.4f} q_crop={sum(qc) / max(1, len(qc)):.4f} "
          f"d_pos={signs['pos']} d_neg={signs['neg']} d_zero={signs['zero']} "
          f"fork_invalid={sum(f['terminal'] == 'invalid' for f in forks)} "
          f"fork_length={sum(f['terminal'] == 'length' for f in forks)} secs={time.time() - t0:.1f}", flush=True)


def install_dataloader_resume():
    """CAPTION_DL_DATALOADER=<global_step_*/dataloader.pt>: continue that run's question order from fresh weights
    (model_path = its HF export). Optimizer/scheduler are not restored (save_model_only); global_step starts at 0."""
    path = os.environ.get("CAPTION_DL_DATALOADER")
    if not path:
        return
    from verl.trainer.ray_trainer import RayPPOTrainer
    if getattr(RayPPOTrainer, "_dl_resume", False):
        return
    original = RayPPOTrainer._load_checkpoint

    def _load_checkpoint(self):
        original(self)
        self.train_dataloader.load_state_dict(torch.load(path, weights_only=False))
        print(f"[caption] dataloader state loaded from {path}; global_step={self.global_step}", flush=True)

    RayPPOTrainer._load_checkpoint, RayPPOTrainer._dl_resume = _load_checkpoint, True


def caption_getitem(self, index):
    """Build dataset and rollout prefixes from the SAME presented original view."""
    from verl.utils import torch_functional as VF
    example = dict(self.dataset[index])
    if example.get("system") != proto.SYSTEM:
        raise ValueError("Dataset SYSTEM differs from frozen v5.1 protocol")
    source_images = example.pop(self.image_key)
    if len(source_images) != 1 or not isinstance(source_images[0], str):
        raise ValueError("Caption training requires one full-resolution original image path")
    image = source_images[0]
    if self.image_dir:
        image = str(Path(self.image_dir) / image)
    env = CENV.CropEnv(image)
    gold = example.get(self.answer_key)
    if isinstance(gold, str) and gold.strip().startswith("{"):
        import ast
        gold = ast.literal_eval(gold)
    gold = (gold.get("ground_truth", "") if isinstance(gold, dict) else gold)
    info = {"head": example.pop("env_head"), "tail": example.pop("env_tail"),
            "system": example.pop("system"),
            "gold": str(gold).strip().upper()}  # fork scoring target for the decision-state baseline; never shown to the policy
    ids, _, grid = encode_initial(self.processor, env, info["head"], info["tail"],
                                  self.min_pixels, self.max_pixels)
    # Never truncate multimodal prompts through their image spans.
    if len(ids) > self.max_prompt_length:
        raise ValueError(f"Caption prompt exceeds budget: {len(ids)} > {self.max_prompt_length}")
    positions = positions_for_sequence(self.processor, ids, grid, axes=4)
    token_ids, attention, positions = VF.postprocess_data(
        input_ids=torch.tensor(ids), attention_mask=torch.ones(len(ids), dtype=torch.long),
        position_ids=positions, max_length=self.max_prompt_length,
        pad_token_id=self.tokenizer.pad_token_id, left_pad=True, truncation="error")
    example.pop(self.prompt_key, None)
    example.update(input_ids=token_ids, attention_mask=attention, position_ids=positions,
                   raw_prompt_ids=raw_image_prompt(ids, grid, self.processor),
                   multi_modal_data={"images": [image], "env": info},
                   ground_truth=example.pop(self.answer_key))
    return example


def install():
    """Install once in driver AND Ray actor/reward workers; no protocol rewrite."""
    from verl.utils.dataset import RLHFDataset
    from verl.workers.rollout.vllm_rollout_spmd import vLLMRollout
    from verl.workers.fsdp_workers import FSDPWorker
    from verl.workers.reward.function import AutoRewardManager
    if getattr(vLLMRollout, "_caption_contracts_installed", False):
        return True
    if os.getenv("CAPTION_DIAG_DIR"):
        from caption_diagnostics import install as install_diagnostics
        install_diagnostics()
    original_init = vLLMRollout.__init__

    def rollout_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._caption_processor = kwargs.get("processor") or (args[3] if len(args) > 3 else None)
        if self._caption_processor is None:
            raise ValueError("Caption rollout requires the training processor")
        engine = self.inference_engine.llm_engine
        model_config = getattr(engine, "model_config", None)
        if model_config is None:
            model_config = engine.vllm_config.model_config
        if model_config.hf_config.model_type != "qwen3_vl":
            raise ValueError("This adapter is scoped to Qwen3-VL")
        contract = vision_contract(self._caption_processor, model_config.hf_config)
        from vllm.multimodal import MULTIMODAL_REGISTRY
        self._caption_mm_processor = MULTIMODAL_REGISTRY.create_processor(
            model_config, tokenizer=self._caption_processor.tokenizer)
        print(f"[caption] vision_contract={contract}", flush=True)

    RLHFDataset.__getitem__ = caption_getitem
    overrides = _engine_overrides()
    if overrides:
        import verl.workers.rollout.vllm_rollout_spmd as spmd
        if getattr(spmd.LLM, "_caption_engine_overrides", None) != tuple(sorted(overrides)):
            real_llm = getattr(spmd.LLM, "_caption_real_llm", None) or spmd.LLM

            def llm_with_overrides(*args, **kwargs):
                kwargs.update(overrides)
                return real_llm(*args, **kwargs)

            llm_with_overrides._caption_real_llm = real_llm
            llm_with_overrides._caption_engine_overrides = tuple(sorted(overrides))
            spmd.LLM = llm_with_overrides
        print(f"[caption] engine overrides={overrides}", flush=True)
    vLLMRollout.__init__ = rollout_init
    vLLMRollout.generate_sequences = generate_sequences_multiturn
    FSDPWorker._process_multi_modal_inputs = actor_multimodal_inputs
    # The original manager assumes a contiguous prefix mask and truncates the answer.
    AutoRewardManager.compute_reward = assistant_reward_batch
    install_std_floor()
    DL.install_actor_patch()
    if DL.beta() > 0:
        def mm_inputs_with_aux(worker, data):
            actor_multimodal_inputs(worker, data)
            DL.build_aux_records(worker, data)
        FSDPWorker._process_multi_modal_inputs = mm_inputs_with_aux
    install_dataloader_resume()
    print(f"[caption] protocol={proto.PROTOCOL} dl_beta={DL.beta()} dl_k={DL.fork_k()}", flush=True)
    vLLMRollout._caption_contracts_installed = True
    return True


REQUIRED_ENV = ("CAPTION_CROP_LAMBDA", "CAPTION_STD_FLOOR", "CAPTION_ROLLOUT_LOG", "CAPTION_DL_H0", "CAPTION_DL_LOG")


def grpo_std_floor_advantage(token_level_rewards, response_mask, index, floor: float):
    """EasyR1 GRPO advantage with the group std floored: (r - mean) / max(std, floor).

    With binary correctness and n=16 any group that is not all-equal has std >= 0.25,
    so floor=0.25 changes only all-correct groups, where the crop cost would otherwise
    be rescaled to unit strength (the coefficient cancels under plain std division).
    """
    from collections import defaultdict
    scores = token_level_rewards.sum(dim=-1)
    id2score = defaultdict(list)
    for i in range(scores.shape[0]):
        id2score[index[i]].append(scores[i])
    id2mean, id2den, floored = {}, {}, set()
    for idx, group in id2score.items():
        assert len(group) > 1, "GRPO needs rollout.n > 1."
        group = torch.tensor(group)
        id2mean[idx] = torch.mean(group)
        std = torch.std(group)
        id2den[idx] = torch.clamp(std, min=floor)
        if std < floor:
            floored.add(idx)
    adv = torch.stack([(scores[i] - id2mean[index[i]]) / id2den[index[i]] for i in range(scores.shape[0])])
    max_floored = max((abs(float(adv[i])) for i in range(len(adv)) if index[i] in floored), default=0.0)
    print(f"[floor] groups={len(id2score)} floored={len(floored)} max_abs_adv_floored={max_floored:.4f}", flush=True)
    returns = adv.unsqueeze(-1) * response_mask
    return returns, returns


def install_std_floor():
    missing = [k for k in REQUIRED_ENV if not os.environ.get(k)]
    if missing:
        raise RuntimeError(f"cropfloor requires env {missing}")
    from verl.trainer import core_algos
    floor = float(os.environ["CAPTION_STD_FLOOR"])

    def grpo(token_level_rewards, response_mask, index, eps: float = 1e-6, **kwargs):
        return grpo_std_floor_advantage(token_level_rewards, response_mask, index, floor)

    core_algos.register_adv_estimator(core_algos.AdvantageEstimator.GRPO)(grpo)
    print(f"[caption] grpo std floor = {floor} lambda = {os.environ['CAPTION_CROP_LAMBDA']}", flush=True)


@torch.no_grad()
def generate_sequences_multiturn(self, prompts):
    from verl.utils.dataset import process_image
    from tensordict import TensorDict
    from verl.protocol import DataProto

    proc = self._caption_processor
    n, base_sampling = sampling_for_group(self.sampling_params, prompts.meta_info)
    input_ids = prompts.batch["input_ids"].repeat_interleave(n, dim=0)
    attention = prompts.batch["attention_mask"].repeat_interleave(n, dim=0)
    original_positions = prompts.batch["position_ids"].repeat_interleave(n, dim=0)
    if original_positions.ndim != 3 or original_positions.shape[1] not in (3, 4):
        raise ValueError("Missing Qwen3-VL mRoPE axes in dataset")
    axes = original_positions.shape[1]
    source_mm = [mm for mm in prompts.non_tensor_batch["multi_modal_data"] for _ in range(n)]
    min_px, max_px = prompts.meta_info["min_pixels"], prompts.meta_info["max_pixels"]
    response_length = int(self.config.response_length)
    context_length = int(getattr(self.config, "max_model_len", None) or
                         (input_ids.shape[-1] + response_length))
    states = []
    for i, mm in enumerate(source_mm):
        env = CENV.CropEnv(mm["images"][0])
        info = mm["env"]
        ids, images, grid = encode_initial(proc, env, info["head"], info["tail"], min_px, max_px)
        supplied = input_ids[i][attention[i].bool()].detach().cpu().tolist()
        if ids != supplied:
            raise ValueError("Dataset/rollout prefix mismatch; rebuild data with the installed adapter")
        states.append({"env": env, "ids": ids, "images": images, "grid": grid, "segments": [],
                       "initial_length": len(ids), "cap": min(len(ids) + response_length, context_length),
                       "terminal": "no_answer", "snaps": []})
    active = list(range(len(states)))
    for round_index in range(CENV.MAX_CROPS + 1):
        if not active:
            break
        requests, samplings, scheduled = [], [], []
        global seq_counter
        seq_counter = getattr(generate_sequences_multiturn, "_seq", 0)
        for i in active:
            state = states[i]
            remaining = state["cap"] - len(state["ids"])
            if remaining <= 0:
                state["terminal"] = "context_limit"
                continue
            requests.append(vllm_preflight(self._caption_mm_processor, state["ids"], state["images"],
                                           state["grid"], proc))
            # decision state s_k (k = round_index + 1) before the turn is sampled: kept for answer-now forks
            state["snaps"].append((round_index, len(state["ids"]), len(state["images"]), len(state["grid"])))
            sampling = copy.deepcopy(base_sampling)
            sampling.seed = ((self.config.seed or 42) * 1000003 + seq_counter) % (2**31)  # 组内不重复且可复现
            seq_counter += 1
            sampling.max_tokens = min(TURN_MAX_TOKENS, remaining)
            samplings.append(sampling)
            scheduled.append(i)
        if not scheduled:
            break
        if os.getenv("CAPTION_DIAG_DIR"):
            from caption_diagnostics import emit
            emit("rollout_turn", round_index=round_index, trajectory_indices=scheduled,
                 prompt_lengths=[len(states[i]["ids"]) for i in scheduled],
                 image_counts=[len(states[i]["images"]) for i in scheduled],
                 grids=[states[i]["grid"] for i in scheduled],
                 generation_budgets=[s.max_tokens for s in samplings])
        outputs = self.inference_engine.generate(prompts=requests, sampling_params=samplings, use_tqdm=False)
        if len(outputs) != len(scheduled):
            raise RuntimeError("vLLM output count differs from scheduled trajectories")
        next_active = []
        for i, output in zip(scheduled, outputs):
            state = states[i]
            if len(output.outputs) != 1:
                raise RuntimeError("Group already expanded; engine must generate n=1")
            if list(output.prompt_token_ids) != state["ids"]:
                raise RuntimeError("vLLM actual prompt differs from CPU-validated token ledger")
            completion = output.outputs[0]
            sampled = list(completion.token_ids)
            if len(state["ids"]) + len(sampled) > state["cap"]:
                raise RuntimeError("Engine exceeded the reserved context budget")
            state["segments"].append((sampled, True))
            state["ids"] += sampled
            state.setdefault("kinds", []).append(None)  # filled after parsing (croponly arm needs the action kind)
            if not sampled or completion.finish_reason == "length":
                state["terminal"] = "length"
                continue
            text = proc.tokenizer.decode(sampled, skip_special_tokens=True).strip()
            try:
                parsed = proto.parse_step(text)
                errors = proto.check_step(parsed, expected_step=round_index + 1,
                                          crops_used=state["env"].crops,
                                          available=state["env"].available, budget=CENV.MAX_CROPS)
            except (ValueError, KeyError, TypeError):
                parsed, errors = {}, ["parse_error"]
            state["kinds"][-1] = "invalid" if errors else parsed.get("kind")
            if errors:
                state["terminal"] = "invalid"
                continue
            if parsed.get("kind") == "answer":
                state["terminal"] = "answer"
                continue
            if parsed.get("kind") != "grounding":
                state["terminal"] = "invalid"
                continue
            receipt = state["env"].step(parsed)
            if receipt is None:
                state["terminal"] = "invalid"
                continue
            image = process_image(receipt["image"], min_px, max_px)
            encoded = proc(images=[image], text=[receipt_chunk(proc, receipt, sampled[-1])],
                           add_special_tokens=False, return_tensors="pt")
            receipt_ids = encoded["input_ids"][0].tolist()
            receipt_grid = grid_list(encoded["image_grid_thw"])
            validate_images(receipt_ids, receipt_grid, proc)
            # Add the observation atomically, leaving at least one token for continuation.
            if len(state["ids"]) + len(receipt_ids) + 1 > state["cap"]:
                state["terminal"] = "context_limit"
                continue
            state["segments"].append((receipt_ids, False))
            state["ids"] += receipt_ids
            state["images"].append(image)
            state["grid"] += receipt_grid
            next_active.append(i)
        generate_sequences_multiturn._seq = seq_counter
        active = next_active

    if n > 1 and DL.beta() > 0:  # training groups only; validation (n=1) and the beta=0 control run no forks
        run_dl_forks(self, states, source_mm, proc, base_sampling, n, min_px, max_px)
    size, prompt_width = input_ids.shape
    device = input_ids.device
    responses = torch.full((size, response_length), self.pad_token_id, dtype=torch.long, device=device)
    loss_mask = torch.zeros_like(responses)
    response_attention = torch.zeros_like(responses)
    positions = torch.zeros((size, axes, prompt_width + response_length), dtype=torch.long, device=device)
    final_mm = []
    for i, state in enumerate(states):
        ids, mask = flatten_segments(state["segments"], response_length)
        length = len(ids)
        responses[i, :length] = torch.tensor(ids, dtype=torch.long, device=device)
        loss_mask[i, :length] = torch.tensor(mask, dtype=torch.long, device=device)
        response_attention[i, :length] = 1
        validate_images(state["ids"], state["grid"], proc)
        expected = positions_for_sequence(proc, state["ids"], state["grid"], axes).to(device)
        offset = prompt_width - state["initial_length"]
        positions[i, :, offset:prompt_width + length] = expected
        final_mm.append({"images": state["images"], "caption_grid": state["grid"],
                         "caption_terminal": state["terminal"], "caption_crops": state["env"].crops,
                         "caption_kinds": state.get("kinds", []), "caption_dl": state.get("dl_aux")})
    batch = TensorDict({"prompts": input_ids, "responses": responses,
                        "input_ids": torch.cat((input_ids, responses), dim=-1),
                        "attention_mask": torch.cat((attention, response_attention), dim=-1),
                        "response_mask": loss_mask, "position_ids": positions}, batch_size=size)
    print(f"[caption] trajectories={size} terminal_answers={sum(s['terminal']=='answer' for s in states)} "
          f"invalid={sum(s['terminal']=='invalid' for s in states)} "
          f"length={sum(s['terminal']=='length' for s in states)} "
          f"context_limits={sum(s['terminal']=='context_limit' for s in states)}", flush=True)
    return DataProto(batch=batch, non_tensor_batch={"multi_modal_data": np.array(final_mm, dtype=object)},
                     meta_info=dict(prompts.meta_info))


def main():
    os.environ["CAPTION_GRPO_PATCH"] = "1"
    install()
    import verl.trainer.main as trainer
    trainer.main()


if __name__ == "__main__":
    main()
