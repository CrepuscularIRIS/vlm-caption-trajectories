"""EasyR1 Qwen3-VL caption GRPO adapter; v5.1 environment is unchanged.

Run from the installed EasyR1 checkout with this repository on PYTHONPATH.
Requires the existing Ray-worker hook calling install() in every worker.
GPU integration remains a release gate; see REVIEW_20261012.md.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path

import random

import numpy as np
import torch
import caption_env as CENV
import counter_credit as CC
from reward_caption import correct_one
import caption_v51 as proto
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
    info = {"head": example.pop("env_head"), "tail": example.pop("env_tail"),
            "system": example.pop("system")}
    info["gold"] = CC.gold_letter(example.get(self.answer_key))  # probe target only; never shown to the model
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
    CC.install_advantage_patch()
    vLLMRollout._caption_contracts_installed = True
    return True


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
                       "terminal": "no_answer"})
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
            if errors:
                state["terminal"] = "invalid"
                continue
            if parsed.get("kind") == "answer":
                state["terminal"] = "answer"
                continue
            if parsed.get("kind") != "grounding":
                state["terminal"] = "invalid"
                continue
            before_ids = state["ids"][:len(state["ids"]) - len(sampled)]
            before_images, before_grid = list(state["images"]), [list(r) for r in state["grid"]]
            receipt = state["env"].step(parsed)
            if receipt is None:
                state["terminal"] = "invalid"
                continue
            think_end = [k for k, t in enumerate(sampled) if t == CC.THINK_END_ID]
            state.setdefault("events", []).append({
                "h_ids": before_ids + sampled[:think_end[-1] + 1] if think_end else None,
                "c_ids": sampled[think_end[-1] + 1:] if think_end else None,
                "images": before_images, "grid": before_grid, "receipt": receipt,
                "box": state["env"].boxes[-1]})
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

    if n > 1:  # training groups only; validation (n=1) runs no probes
        run_probes(self, states, source_mm, proc, base_sampling, context_length)

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
        row = {"images": state["images"], "caption_grid": state["grid"],
               "caption_terminal": state["terminal"], "caption_crops": state["env"].crops}
        if "caption_price" in state:
            row.update(caption_price=state["caption_price"], caption_correct=state["caption_correct"],
                       caption_verified=state["caption_verified"], caption_probe=state["caption_probe"])
        final_mm.append(row)
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


def _probe_ids(proc, ev, image, im_end):
    close = "" if ev["c_ids"] and ev["c_ids"][-1] == im_end else "<|im_end|>"
    enc = proc(images=[image], text=[CC.receipt_probe_chunk(ev["receipt"], close)],
               add_special_tokens=False, return_tensors="pt")
    return (ev["h_ids"] + ev["c_ids"] + enc["input_ids"][0].tolist(), ev["images"] + [image],
            ev["grid"] + grid_list(enc["image_grid_thw"]))


def run_probes(self, states, source_mm, proc, base_sampling, context_length):
    """CounterCredit probes for every executed crop of every correct trajectory (paper App. C)."""
    letter_ids = CC.letter_token_ids(proc.tokenizer)
    im_end = proc.tokenizer.convert_tokens_to_ids("<|im_end|>")
    now_ids = proc.tokenizer.encode(CC.now_suffix(), add_special_tokens=False)
    step = getattr(run_probes, "_calls", 0)
    run_probes._calls = step + 1
    requests, keys = [], []
    skipped = 0
    for ti, state in enumerate(states):
        gold = source_mm[ti]["env"].get("gold")
        text = proc.tokenizer.decode([t for seg, a in state["segments"] if a for t in seg], skip_special_tokens=True)
        correct = bool(gold) and state["terminal"] == "answer" and correct_one("caption", text, gold) > 0.5
        state["caption_correct"] = float(correct)
        state["_values"] = [[None, None, []] for _ in state.get("events", [])]
        if not correct:
            continue
        rng = random.Random((self.config.seed or 42) * 7919 + step * 100003 + ti)
        for ei, ev in enumerate(state.get("events", [])):
            if ev["h_ids"] is None:
                skipped += 1
                continue
            variants = [("now", ev["h_ids"] + now_ids, ev["images"], ev["grid"]),
                        ("real",) + _probe_ids(proc, ev, ev["receipt"]["image"], im_end)]
            for _ in range(CC.K):
                variants.append(("rand",) + _probe_ids(proc, ev, CC.random_patch(state["env"], ev["box"], rng), im_end))
            for kind, ids, images, grid in variants:
                if len(ids) + 1 > context_length:
                    skipped += 1
                    continue
                try:
                    requests.append(vllm_preflight(self._caption_mm_processor, ids, images, grid, proc))
                except ValueError as err:
                    print(f"[counter_credit] probe preflight failed: {err}", flush=True)
                    skipped += 1
                    continue
                keys.append((ti, ei, kind, gold))
    if requests:
        sampling = copy.deepcopy(base_sampling)
        sampling.n, sampling.max_tokens, sampling.logprobs = 1, 1, 20
        sampling.temperature, sampling.top_p, sampling.top_k = 1.0, 1.0, -1
        sampling.seed = (self.config.seed or 42) * 31 + step
        outputs = self.inference_engine.generate(prompts=requests, sampling_params=[sampling] * len(requests),
                                                 use_tqdm=False)
        for (ti, ei, kind, gold), out in zip(keys, outputs):
            top = {tid: lp.logprob for tid, lp in out.outputs[0].logprobs[0].items()}
            u = CC.score_from_logprobs(top, letter_ids, gold)
            slot = states[ti]["_values"][ei]
            if kind == "now":
                slot[0] = u
            elif kind == "real":
                slot[1] = u
            else:
                slot[2].append(u)
    verified_total = 0
    for state in states:
        values = [CC.crop_values(*v) for v in state["_values"]]
        priced = CC.trajectory_price(state["caption_correct"] > 0.5, state["env"].crops, values)
        state["caption_price"] = priced["price"]
        state["caption_verified"] = priced["n_verified"]
        state["caption_probe"] = [{k: (round(x, 4) if isinstance(x, float) else x) for k, x in v.items()
                                   if k in ("eligible", "u_now", "u_real", "u_rand", "D", "E", "q")} for v in values]
        verified_total += priced["n_verified"]
        state.pop("_values", None)
        state.pop("events", None)
    print(f"[counter_credit] probes={len(requests)} skipped={skipped} verified={verified_total} "
          f"correct={sum(s['caption_correct'] > 0.5 for s in states)}/{len(states)}", flush=True)


def main():
    os.environ["CAPTION_GRPO_PATCH"] = "1"
    install()
    import verl.trainer.main as trainer
    trainer.main()


if __name__ == "__main__":
    main()
