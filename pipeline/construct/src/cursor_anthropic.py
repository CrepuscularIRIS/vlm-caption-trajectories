"""Grok 4.7 through CLIProxy's Cursor provider on the Anthropic /v1/messages route (2026-10-05, user: use Cursor).

Why not the OpenAI route: CLIProxy's Cursor provider on /v1/chat/completions writes each image to
~/.cursor/projects/.../assets/image-N.jfif and fails with 'file exists' on the next request (then cools the auth).
The /v1/messages route is what LiteLLM's working 'grok-4.7' entry uses. Model id cursor/grok-4.7-low = same model
Grok 4.7 at low effort (the protocol effort). Receipt / resume contract mirrors glm_direct.GLMDirectAPI.
"""
import fcntl
import json
import time

import requests
from dotenv import dotenv_values

from probe_api import digest
from trajgen_models import RoleAPI

PROXY_ENV = '${PROXY_DIR}/.env'
UPSTREAM = 'cursor/grok-4.7-low'
STOP = {'end_turn': 'stop', 'stop_sequence': 'stop', 'max_tokens': 'length'}


def to_anthropic(msgs: list[dict]) -> tuple[str, list[dict]]:
    system, out = [], []
    for m in msgs:
        parts = m['content'] if isinstance(m['content'], list) else [{'type': 'text', 'text': m['content']}]
        if m['role'] == 'system':
            system += [p['text'] for p in parts if p.get('type') == 'text']
            continue
        blocks = []
        for p in parts:
            if p.get('type') == 'text':
                blocks.append({'type': 'text', 'text': p['text']})
            elif p.get('type') == 'image_url':
                head, data = p['image_url']['url'].split(',', 1)
                blocks.append({'type': 'image', 'source': {'type': 'base64', 'media_type': head[5:].split(';')[0], 'data': data}})
        out.append({'role': m['role'], 'content': blocks})
    return '\n\n'.join(system), out


class CursorAnthropicAPI(RoleAPI):
    def __init__(self, out, model='grok-4.7-cur', rpm=120, max_tokens=8192):
        super().__init__(out, model, rpm=rpm, max_tokens=max_tokens)
        env = dotenv_values(PROXY_ENV)
        self.url = env['CLIPROXY_ANTHROPIC_BASE'].rstrip('/') + '/v1/messages'
        self.secret = env['CLIPROXY_API_KEY']

    def call(self, label, msgs, max_tokens=None, *, replicate_id, temperature=0., top_p=None):
        safe, images = self.safe_messages(msgs)
        system, amsgs = to_anthropic(msgs)
        payload = {'model': UPSTREAM, 'messages': amsgs, 'max_tokens': max_tokens or self.max_tokens,
                   'temperature': temperature}
        if system:
            payload['system'] = system
        signature = {'model': UPSTREAM, 'messages': safe, 'max_tokens': payload['max_tokens'], 'temperature': temperature,
                     'replicate_id': replicate_id, 'route': 'cliproxy-anthropic-cursor'}
        key = digest(signature)
        with (self.out / 'locks' / (key + '.lock')).open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            dest = self.out / 'calls' / (key + '.json')
            if dest.exists():
                old = json.loads(dest.read_text())
                if old.get('finish_reason') not in ('error', None):
                    return old
            errors, result, start = [], None, time.monotonic()
            for attempt in range(4):
                self.limiter.wait()
                try:
                    r = requests.post(self.url, json=payload, timeout=(20, 300),
                                      headers={'x-api-key': self.secret, 'Authorization': 'Bearer ' + self.secret,
                                               'anthropic-version': '2023-06-01'})
                    if r.status_code != 200:
                        errors.append({'status': r.status_code, 'attempt': attempt + 1,
                                       'detail': r.text.replace(self.secret, '[REDACTED]')[:600]})
                        if r.status_code in (400, 404, 413):
                            break
                        time.sleep(5 * (attempt + 1))
                        continue
                    data = r.json()
                    text = ''.join(b.get('text', '') for b in data.get('content', []) if b.get('type') == 'text')
                    thinking = ''.join(b.get('thinking', '') for b in data.get('content', []) if b.get('type') == 'thinking')
                    u = data.get('usage') or {}
                    result = {'content': text, 'reasoning_content': thinking,
                              'finish_reason': STOP.get(data.get('stop_reason'), data.get('stop_reason')),
                              'returned_model': data.get('model'), 'response_id': data.get('id'),
                              'usage': {'prompt_tokens': u.get('input_tokens'), 'completion_tokens': u.get('output_tokens'), **u}}
                    break
                except (requests.RequestException, ValueError, KeyError) as exc:
                    errors.append({'error_type': type(exc).__name__, 'attempt': attempt + 1})
                    time.sleep(5 * (attempt + 1))
            if result is None:
                result = {'content': '', 'reasoning_content': '', 'finish_reason': 'error', 'returned_model': None, 'usage': {}}
            result.update(label=label, request=signature, request_sha256=key, replicate_id=replicate_id, images=images,
                          errors=errors, route='cliproxy-anthropic-cursor', elapsed_s=round(time.monotonic() - start, 3))
            if dest.exists():
                dest.with_name(dest.stem + '.attempt-' + str(time.time_ns()) + '.json').write_bytes(dest.read_bytes())
            tmp = dest.with_suffix('.tmp')
            tmp.write_text(json.dumps(result, ensure_ascii=False))
            tmp.replace(dest)
            return result
