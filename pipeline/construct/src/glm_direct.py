"""Direct Zhipu route for glm-5.3-flash (2026-10-03, user decision: no LiteLLM for GLM).

LiteLLM routed the three Zhipu keys by `order` priority, so nearly all traffic hit one key and Zhipu
returned rate_limit_error 1302. Here the coding-plan endpoint is called directly with the user's two
priority keys: ZHIPU_API_KEY2 (Max plan) and ZHIPU_API_KEY0 (Pro plan), weighted 2:1 toward Max.
A key that returns 429/5xx cools down briefly and the next attempt uses another key; a key that returns
401/403 is disabled for this process. Same receipt/resume contract as RoleAPI (request hash excludes the key).
The receipt records the route and key slot (slot name only, never the key).
Effort: Zhipu has no effort scale; 'high'/'medium' -> thinking enabled, 'low' -> thinking disabled.
"""
import fcntl
import json
import re
import threading
import time
from datetime import datetime, timedelta, timezone

import requests
from dotenv import dotenv_values

from probe_api import digest
from trajgen_models import RoleAPI

URL = 'https://open.bigmodel.cn/api/coding/paas/v4/chat/completions'
PROXY_ENV = '${PROXY_DIR}/.env'
# 2026-10-03: all three keys. User 08:15: KEY0 is a Pro plan, KEY1 and KEY2 are Max; KEY1 showed no 429 under load.
SLOTS = (('ZHIPU_API_KEY1', 3), ('ZHIPU_API_KEY2', 3), ('ZHIPU_API_KEY0', 1))  # (env var, weight)
COOL_S = 20
# Per-key in-flight cap (2026-10-04 14:45): 1302 '账户已达到速率限制' bursts came from round-robin sending to a key
# regardless of how many of its requests were open. A worker now takes the least-loaded key below its cap and waits
# if all are full, so the job's worker count can exceed what one key tolerates without trading progress for 429s.
INFLIGHT_CAP = {'ZHIPU_API_KEY1': 3, 'ZHIPU_API_KEY2': 5, 'ZHIPU_API_KEY0': 3}  # 21:10 KEY1 alone drew 21 x 1302: 5 -> 3
# Zhipu code 1308 = per-key 5-hour usage cap ('...将在 2026-10-04 17:04:35 重置', Beijing time): cool that key until the
# stated reset instead of retrying it every COOL_S (2026-10-04: KEY1 drew 111 such 429s in 40 min).
QUOTA_FALLBACK_S, QUOTA_MAX_S = 3600, 5 * 3600
RESET_RE = re.compile(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) 重置')


def quota_cool_s(detail: str) -> float | None:
    if '"1308"' not in detail:
        return None
    m = RESET_RE.search(detail)
    if not m:
        return QUOTA_FALLBACK_S
    reset = datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone(timedelta(hours=8)))
    return min(QUOTA_MAX_S, max(COOL_S, (reset - datetime.now(timezone.utc)).total_seconds() + 30))


class KeyPool:
    def __init__(self, env_path: str = PROXY_ENV):
        env = dotenv_values(env_path)
        self.keys = {name: env[name] for name, _ in SLOTS if env.get(name)}
        if not self.keys:
            raise KeyError('no Zhipu priority key found in ' + env_path)
        self.order = [name for name, w in SLOTS if name in self.keys for _ in range(w)]
        self.i, self.cool, self.dead = 0, {}, set()
        self.inflight = {n: 0 for n in self.keys}
        self.rank = {n: i for i, (n, _) in enumerate(SLOTS)}
        self.lock = threading.Lock()

    def take(self) -> tuple[str, str] | None:
        while True:
            with self.lock:
                live = [n for n in self.keys if n not in self.dead]
                if not live:
                    return None
                t = time.monotonic()
                ready = [n for n in live if self.cool.get(n, 0) <= t and self.inflight[n] < INFLIGHT_CAP.get(n, 4)]
                if ready:
                    name = min(ready, key=lambda n: (self.inflight[n] / INFLIGHT_CAP.get(n, 4), self.rank[n]))
                    self.inflight[name] += 1
                    return name, self.keys[name]
                wait = min([self.cool[n] - t for n in live if self.cool.get(n, 0) > t] + [0.5])
            time.sleep(max(0.05, min(wait, 0.5)))

    def release(self, name: str) -> None:
        with self.lock:
            self.inflight[name] = max(0, self.inflight[name] - 1)

    def penalize(self, name: str, status: int, detail: str = '') -> None:
        with self.lock:
            if status in (401, 403):
                self.dead.add(name)
            else:
                self.cool[name] = time.monotonic() + (quota_cool_s(detail) or COOL_S)


class GLMDirectAPI(RoleAPI):
    def __init__(self, out, model='glm-5.3-flash', rpm=120, max_tokens=16384, effort='high'):
        super().__init__(out, model, rpm=rpm, max_tokens=max_tokens)
        self.pool = KeyPool()
        self.thinking = 'disabled' if effort == 'low' else 'enabled'

    def call(self, label, msgs, max_tokens=None, *, replicate_id, temperature=0., top_p=None):
        safe, images = self.safe_messages(msgs)
        payload = {'model': self.model, 'messages': msgs, 'temperature': temperature,
                   'max_tokens': max_tokens or self.max_tokens, 'thinking': {'type': self.thinking}}
        if top_p is not None:
            payload['top_p'] = top_p
        signature = {**payload, 'messages': safe, 'replicate_id': replicate_id, 'route': 'zhipu-coding-direct'}
        key = digest(signature)
        with (self.out / 'locks' / (key + '.lock')).open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            dest = self.out / 'calls' / (key + '.json')
            if dest.exists():
                old = json.loads(dest.read_text())
                if old.get('finish_reason') not in ('error', None):
                    return old
            errors, result, start, slot = [], None, time.monotonic(), None
            for attempt in range(5):
                got = self.pool.take()
                if got is None:
                    self.blocked.set()
                    raise RuntimeError('Authentication/quota circuit open: all Zhipu keys disabled')
                slot, secret = got
                self.limiter.wait()
                try:
                    try:
                        r = requests.post(URL, json=payload, headers={'Authorization': 'Bearer ' + secret}, timeout=(20, 600))
                    finally:
                        self.pool.release(slot)
                    if r.status_code != 200:
                        errors.append({'status': r.status_code, 'attempt': attempt + 1, 'key_slot': slot,
                                       'detail': r.text.replace(secret, '[REDACTED]')[:600]})
                        self.pool.penalize(slot, r.status_code, r.text)
                        if r.status_code in (400, 404, 413):
                            break
                        continue
                    data = r.json()
                    choice = data['choices'][0]
                    m = choice['message']
                    result = {'content': m.get('content') or '', 'reasoning_content': m.get('reasoning_content') or '',
                              'finish_reason': choice.get('finish_reason'), 'returned_model': data.get('model'),
                              'usage': data.get('usage', {}), 'response_id': data.get('id')}
                    break
                except (requests.RequestException, ValueError, KeyError, IndexError) as exc:
                    errors.append({'error_type': type(exc).__name__, 'attempt': attempt + 1, 'key_slot': slot})
                    self.pool.penalize(slot, 0)
            if result is None:
                result = {'content': '', 'reasoning_content': '', 'finish_reason': 'error', 'returned_model': None, 'usage': {}}
            result.update(label=label, request=signature, request_sha256=key, replicate_id=replicate_id, images=images,
                          errors=errors, route='zhipu-coding-direct', key_slot=slot,
                          elapsed_s=round(time.monotonic() - start, 3))
            if dest.exists():
                dest.with_name(dest.stem + '.attempt-' + str(time.time_ns()) + '.json').write_bytes(dest.read_bytes())
            tmp = dest.with_suffix('.tmp')
            tmp.write_text(json.dumps(result, ensure_ascii=False))
            tmp.replace(dest)
            return result
