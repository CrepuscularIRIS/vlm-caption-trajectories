"""Blind fact reader over an OpenAI-compatible API (2026-10-03: Grok replaces Gemini/agy, whose quota is unstable).

Same interface as CCReviewer / AgyReader: call(label, system, prompt, views) -> receipt with 'valid', 'text',
'request_sha256'. The request carries ONLY the system text, the neutral prompt and the listed view images (base64);
nothing else from the trajectory. Valid = HTTP 200, non-empty text and a returned model that starts with the requested
family prefix. One retry on a non-200 / empty answer; every attempt is reserved in the durable ledger first.
"""
import base64
import hashlib
import json
import time
from pathlib import Path

import requests
from dotenv import dotenv_values

import models_v51
from probe_api import digest
from tiered_transport import write_json

PREFIX = {'grok-4.7': 'grok', 'gpt-6.1-sol': 'gpt-6.1-sol'}


class ApiReader:
    def __init__(self, out, ledger, model='grok-4.7', max_tokens=2000, effort='low'):
        self.out, self.ledger, self.model = Path(out), ledger, model
        self.out.mkdir(parents=True, exist_ok=True)
        ep = models_v51.tm.MODELS[model]
        self.url = ep.base + '/v1/chat/completions'
        self.key = dotenv_values(ep.key_file).get(ep.key_var)
        self.max_tokens, self.effort = max_tokens, effort

    def call(self, label, system, prompt, views):
        image_meta = [dict(view_id=v['view_id'], sha256=hashlib.sha256(Path(v['path']).read_bytes()).hexdigest())
                      for v in views]
        key = digest(dict(model=self.model, system=system, prompt=prompt, images=image_meta, label=label, route='api'))
        dest = self.out / (key + '.json')
        if dest.exists():
            cached = json.loads(dest.read_text())
            if cached.get('valid'):
                return cached
        content = [dict(type='text', text=prompt)]
        for v in views:
            data = Path(v['path']).read_bytes()
            mime = 'image/png' if data[:4] == b'\x89PNG' else 'image/jpeg'
            content += [dict(type='text', text=f"[{v['view_id']}]"),
                        dict(type='image_url', image_url=dict(url=f'data:{mime};base64,' + base64.b64encode(data).decode()))]
        body = dict(model=self.model, max_tokens=self.max_tokens, reasoning_effort=self.effort,
                    messages=[dict(role='system', content=system), dict(role='user', content=content)])
        receipt, start = None, time.monotonic()
        for attempt in (1, 2):
            self.ledger.reserve(self.model, key, label, request_limit=2)
            try:
                r = requests.post(self.url, json=body, headers={'Authorization': 'Bearer ' + self.key}, timeout=(20, 300))
                ok = r.status_code == 200
                j = r.json() if ok else {}
                text = (j.get('choices') or [{}])[0].get('message', {}).get('content') or '' if ok else ''
                returned = j.get('model') if ok else None
                receipt = dict(status=r.status_code, text=text, returned_models=[returned] if returned else [],
                               detail=None if ok else r.text.replace(self.key, '[REDACTED]')[:400], usage=j.get('usage'))
            except (requests.RequestException, ValueError) as exc:
                receipt = dict(status=None, text='', returned_models=[], detail=f'{type(exc).__name__}')
            receipt['valid'] = bool(receipt['status'] == 200 and receipt['text'].strip() and receipt['returned_models']
                                    and receipt['returned_models'][0].startswith(PREFIX.get(self.model, self.model)))
            if receipt['valid']:
                break
            time.sleep(5)
        receipt.update(request_sha256=key, model=self.model, route='api ' + self.url.split('/v1')[0], label=label,
                       images=image_meta, attempts=attempt, elapsed_s=round(time.monotonic() - start, 3))
        write_json(dest, receipt)
        return receipt
