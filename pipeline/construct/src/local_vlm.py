"""Local open VLM generators (Qwen2.5-VL / Qwen3-VL) behind the same call() contract as RoleAPI.

GPU1 only (project rule). Receipts are written to <out>/calls/<sha>.json exactly like the API
transport, so resume, auditing and analysis code work unchanged. Sampling is seeded from the
request hash, so a repeated request reproduces the same output.
"""
import hashlib
import json
import os
import threading
import time
from pathlib import Path

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '1')
os.environ.setdefault('HF_HOME', '${HF_HOME}')
os.environ.setdefault('HF_HUB_OFFLINE', '1')

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from probe_api import digest


class LocalVLM:
    def __init__(self, out: Path, model: str, path: str, max_tokens: int = 2048, split_thinking: bool | None = None):
        assert os.environ['CUDA_VISIBLE_DEVICES'] == '1', 'GPU0 is forbidden'
        self.out, self.model, self.max_tokens = Path(out), model, max_tokens
        self.split_thinking = ('thinking' in model.lower()) if split_thinking is None else split_thinking
        for d in ('calls', 'media', 'locks'):
            (self.out / d).mkdir(parents=True, exist_ok=True)
        self.proc = AutoProcessor.from_pretrained(path)
        self.net = AutoModelForImageTextToText.from_pretrained(
            path, dtype=torch.bfloat16, device_map='cuda:0', attn_implementation='sdpa').eval()
        self.lock = threading.Lock()

    def image(self, path):
        data = Path(path).read_bytes()
        return {'type': 'image', 'path': str(path), 'sha256': hashlib.sha256(data).hexdigest()}

    @staticmethod
    def _convert(msgs):
        conv, safe = [], []
        for m in msgs:
            parts = m['content'] if isinstance(m['content'], list) else [{'type': 'text', 'text': m['content']}]
            conv.append({'role': m['role'], 'content': [
                {'type': 'image', 'path': p['path']} if p['type'] == 'image' else {'type': 'text', 'text': p['text']}
                for p in parts]})
            safe.append({'role': m['role'], 'content': [
                {'type': 'image', 'sha256': p['sha256']} if p['type'] == 'image' else p for p in parts]})
        return conv, safe

    supports_prefill = True

    @property
    def coord_mode(self) -> str:
        """Native grounding convention: Qwen2.5-VL = absolute pixels of the resized model input,
        Qwen3-VL = integers on a 0-1000 relative grid."""
        return 'rel1000' if 'qwen3' in self.model.lower() else 'pixels'

    def model_input_size(self, path: str) -> tuple[int, int]:
        """(width, height) of the image as the model sees it after the processor's resize."""
        ip = self.proc.image_processor
        with Image.open(path) as im:
            grid = ip(images=[im.convert('RGB')], return_tensors='pt')['image_grid_thw'][0].tolist()
        return int(grid[2] * ip.patch_size), int(grid[1] * ip.patch_size)

    def call(self, label, msgs, max_tokens=None, *, replicate_id, temperature=0.7, top_p=0.9, prefill=''):
        """prefill: text forced at the start of the assistant turn (format scaffold); it is prepended to
        the returned content/raw_text so the saved turn is complete."""
        conv, safe = self._convert(msgs)
        max_tokens = max_tokens or self.max_tokens
        signature = dict(model=self.model, messages=safe, temperature=temperature, top_p=top_p,
                         max_tokens=max_tokens, replicate_id=replicate_id)
        if prefill:
            signature['prefill'] = prefill                      # absent key keeps old cache keys valid
        key = digest(signature)
        dest = self.out / 'calls' / f'{key}.json'
        if dest.exists():
            return json.loads(dest.read_text())
        t0 = time.monotonic()
        with self.lock, torch.inference_mode():
            inputs = self.proc.apply_chat_template(conv, tokenize=True, add_generation_prompt=True,
                                                   return_dict=True, return_tensors='pt').to(self.net.device)
            if prefill:
                pre = torch.tensor([self.proc.tokenizer.encode(prefill, add_special_tokens=False)],
                                   device=inputs['input_ids'].device)
                seq = inputs['input_ids'].shape[1]
                for k, v in list(inputs.items()):          # every per-token tensor grows with the prefix
                    if torch.is_tensor(v) and v.dim() == 2 and v.shape[1] == seq:
                        fill = pre if k == 'input_ids' else (torch.ones_like(pre) if k == 'attention_mask'
                                                             else torch.zeros_like(pre))
                        inputs[k] = torch.cat([v, fill.to(v.dtype)], dim=1)
            torch.manual_seed(int(key[:8], 16))
            gen = dict(max_new_tokens=max_tokens, do_sample=temperature > 0)
            if temperature > 0:
                gen.update(temperature=temperature, top_p=top_p or 1.0)
            out = self.net.generate(**inputs, **gen)
        new = out[0, inputs['input_ids'].shape[1]:]
        text = self.proc.decode(new, skip_special_tokens=True)
        text = (prefill + text).strip() if prefill else text.strip()
        eos = self.proc.tokenizer.eos_token_id
        eos = set(eos if isinstance(eos, list) else [eos]) | set(self.net.generation_config.eos_token_id or [])
        finished = len(new) < max_tokens or int(new[-1]) in eos
        raw_text = text
        reasoning = ''
        # Only *Thinking* checkpoints emit hidden reasoning in <think>. For Instruct models the
        # <think> block is the caption we asked for (caption MVP protocol) and must stay in content.
        if self.split_thinking and '</think>' in text:
            reasoning, text = text.split('</think>', 1)
            reasoning, text = reasoning.replace('<think>', '').strip(), text.strip()
        n_in, n_out = int(inputs['input_ids'].shape[1]), int(len(new))
        result = dict(content=text, raw_text=raw_text, reasoning_content=reasoning, finish_reason='stop' if finished else 'length',
                      returned_model=self.model, usage=dict(prompt_tokens=n_in, completion_tokens=n_out,
                                                            total_tokens=n_in + n_out),
                      label=label, request=signature, request_sha256=key, replicate_id=replicate_id,
                      images=[p['sha256'] for m in safe for p in m['content'] if p.get('type') == 'image'],
                      errors=[], elapsed_s=round(time.monotonic() - t0, 3))
        tmp = dest.with_suffix('.tmp')
        tmp.write_text(json.dumps(result, ensure_ascii=False))
        tmp.replace(dest)
        return result
