"""Model registry for trajectory construction roles (generator / monitor / judge).

Endpoints were smoke-verified on 2026-09-30 with an image-reading probe (text "K7Q" +
blue circle): every entry below returned 200 and described the image correctly.
Keys are read from env files at call time and never logged.

- CLIProxy (:8317) exposes subscription-backed models over an OpenAI-compatible API.
  Calls count against the corresponding subscription quota. `claude-opus-5-5` is NOT
  usable here (proxy's Claude Code client version is too old -> 400); use the claude CLI.
- LiteLLM (:4001) currently injects `thinking`/`reasoning_effort` into gpt-6*/claude
  groups and fails for plain chat calls; only GLM routes are used from it.
- Kimi Coding API serves k3-256k directly (OpenAI format on /v1/chat/completions).
  k3-256k rejects temperature != 1 and top_p != 0.95 (400 errors); both are pinned below.
"""
import fcntl
import json
import time
from dataclasses import dataclass
from pathlib import Path

import requests
from dotenv import dotenv_values

from probe_api import digest
from probe_replica_api import ReplicaAPI

REPO_ENV = Path('${ICML_DIR}/.env')
PROXY_ENV = Path('${PROXY_DIR}/.env')


@dataclass(frozen=True)
class Endpoint:
    base: str          # without trailing /v1; transport appends /v1/chat/completions
    key_file: Path
    key_var: str
    fixed_temperature: float | None = None   # provider-enforced; overrides caller's value
    fixed_top_p: float | None = None


CLIPROXY = Endpoint('http://127.0.0.1:8317', PROXY_ENV, 'CLIPROXY_API_KEY')
LITELLM = Endpoint('http://127.0.0.1:4001', REPO_ENV, 'LITELLM_API_KEY')
KIMI = Endpoint('https://api.kimi.com/coding', PROXY_ENV, 'KIMI_API_KEY0', fixed_temperature=1.0,
                fixed_top_p=0.95)

MODELS = {
    'gpt-6.1-sol': CLIPROXY,
    'gpt-6-sol': CLIPROXY,
    'gemini-3.8-flash-high': CLIPROXY,   # returned_model reports gemini-3.8-flash; API route w/o agy
    'gemini-3.1-pro-low': CLIPROXY,
    'claude-sonnet-5-5': CLIPROXY,       # direct works (OpenAI + Anthropic formats); opus-5-5 does not
    'gpt-6-astra': CLIPROXY,
    'gemini-3.1-pro-low': CLIPROXY,
    'grok-4.7': CLIPROXY,          # returned_model reports grok-4.7-build
    'k3-256k': KIMI,
    'glm-5.3': LITELLM,
    'glm-5.3-flash': LITELLM,
}

# CLI-only models (no working direct route as of 2026-09-30), via cli_backends.cli_call.
# claude-opus-5-5: CLIProxy/LiteLLM both 400 (proxy's Claude Code client 2.1.63 < 2.1.280 required).
# gemini-3.1-pro-high: not exposed by CLIProxy (only -low is).
# agy: system prompt must forbid non-view tools (gemini-3.8-flash once ran a command, empty reply).
CLI_MODELS = {
    'claude-sonnet-5': 'claude',        # user choice: CLI (direct CLIProxy also worked once); ~7.5 s/call
    'claude-opus-5-5': 'claude',        # ~7-12 s/call
    'gemini-3.1-pro-high': 'agy',       # ~20-28 s/call
}

# Reasoning effort (user decision 2026-09-30): default medium, low allowed.
# API: `reasoning_effort` is accepted by Sol/K3/Gemini/Sonnet routes, but its effect is unverified
# (Sol reported 0 reasoning tokens at both levels). CLI: claude --effort; agy --effort or -medium/-low ids.
DEFAULT_EFFORT = 'medium'

# Roles (user decision 2026-09-30, supersedes earlier assignments):
# generators = local open models on GPU1; a pool of API monitors drags them step by step;
# only very hard cases escalate to strong models for a second judgment.
GENERATORS_LOCAL = {
    'qwen2.5-vl-7b': '${MODEL_DIR}/Qwen2.5-VL-7B-Instruct',
    'qwen3-vl-8b': 'Qwen/Qwen3-VL-8B-Instruct',             # HF cache, 4/4 shards complete
    'qwen3-vl-8b-thinking': 'Qwen/Qwen3-VL-8B-Thinking',    # HF cache, 4/4 shards complete
}
MONITORS = ['claude-sonnet-5', 'gpt-6.1-sol', 'grok-4.7', 'k3-256k', 'gemini-3.8-flash-high']
ESCALATION = {'gpt-6-astra': 'api', 'claude-opus-5-5': 'claude-cli'}
GENERATOR = None
JUDGE = 'glm-5.3-flash'

class RoleAPI(ReplicaAPI):
    """ReplicaAPI plus provider extras (reasoning_effort, pinned temperature) and error detail.

    Same receipt/resume contract as ReplicaAPI.call: request hash covers the full payload
    (images replaced by sha256), completed receipts are reused, failed attempts are kept.
    """

    extra: dict = {}
    pinned_temperature: float | None = None
    pinned_top_p: float | None = None

    def call(self, label, msgs, max_tokens=None, *, replicate_id, temperature=0., top_p=None):
        if self.pinned_temperature is not None:
            temperature = self.pinned_temperature
        if self.pinned_top_p is not None:
            top_p = self.pinned_top_p
        safe, images = self.safe_messages(msgs)
        payload = {'model': self.model, 'messages': msgs, 'temperature': temperature,
                   'max_tokens': max_tokens or self.max_tokens, **self.extra}
        if top_p is not None:
            payload['top_p'] = top_p
        signature = {**payload, 'messages': safe, 'replicate_id': replicate_id}
        key = digest(signature)
        with (self.out / 'locks' / (key + '.lock')).open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            dest = self.out / 'calls' / (key + '.json')
            if dest.exists():
                old = json.loads(dest.read_text())
                if old.get('finish_reason') not in ('error', None):
                    return old
            if self.blocked.is_set():
                raise RuntimeError('Authentication/quota circuit open')
            errors, result, start = [], None, time.monotonic()
            for attempt in range(4):
                self.limiter.wait()
                try:
                    r = requests.post(self.base + '/v1/chat/completions', json=payload,
                                      headers={'Authorization': 'Bearer ' + self.key}, timeout=(20, 300))
                    if r.status_code != 200:
                        errors.append({'status': r.status_code, 'attempt': attempt + 1,
                                       'detail': r.text.replace(self.key, '[REDACTED]')[:800]})
                        if r.status_code in (401, 402, 403):
                            self.blocked.set()
                        if r.status_code in (400, 401, 402, 403, 404, 413):
                            break
                        time.sleep(min(30, 2 ** (attempt + 1)))
                        continue
                    data = r.json()
                    choice = data['choices'][0]
                    m = choice['message']
                    result = {'content': m.get('content') or '', 'reasoning_content': m.get('reasoning_content') or '',
                              'finish_reason': choice.get('finish_reason'), 'returned_model': data.get('model'),
                              'usage': data.get('usage', {}), 'response_id': data.get('id')}
                    break
                except (requests.RequestException, ValueError, KeyError, IndexError) as exc:
                    errors.append({'error_type': type(exc).__name__, 'attempt': attempt + 1})
                    time.sleep(min(20, 2 ** attempt))
            if result is None:
                result = {'content': '', 'reasoning_content': '', 'finish_reason': 'error',
                          'returned_model': None, 'usage': {}}
            result.update(label=label, request=signature, request_sha256=key, replicate_id=replicate_id,
                          images=images, errors=errors, elapsed_s=round(time.monotonic() - start, 3))
            if dest.exists():
                dest.with_name(dest.stem + '.attempt-' + str(time.time_ns()) + '.json').write_bytes(dest.read_bytes())
            tmp = dest.with_suffix('.tmp')
            tmp.write_text(json.dumps(result, ensure_ascii=False))
            tmp.replace(dest)
            return result


def make_api(model: str, out: Path, rpm: int = 30, max_tokens: int = 8192,
             effort: str | None = DEFAULT_EFFORT) -> RoleAPI:
    """RoleAPI bound to the verified endpoint for `model`. effort=None sends no reasoning_effort."""
    if model in GENERATORS_LOCAL:
        from local_vlm import LocalVLM   # imports torch; only when a local generator is requested
        return LocalVLM(out, model, GENERATORS_LOCAL[model], max_tokens=min(max_tokens, 4096))
    if model not in MODELS:
        raise KeyError(f'unregistered model {model!r}; verify it with an image probe first')
    ep = MODELS[model]
    key = dotenv_values(ep.key_file).get(ep.key_var)
    if not key:
        raise KeyError(f'{ep.key_var} missing in {ep.key_file}')
    api = RoleAPI(out, model, rpm=rpm, max_tokens=max_tokens)
    api.base, api.key = ep.base, key
    api.pinned_temperature = ep.fixed_temperature
    api.pinned_top_p = ep.fixed_top_p
    api.extra = {'reasoning_effort': effort} if effort else {}
    return api
