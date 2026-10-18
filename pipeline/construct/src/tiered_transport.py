"""Bounded transports for binding-tiered-v1. No CUDA; credentials never enter receipts."""
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path

import requests
from dotenv import dotenv_values

from probe_api import digest
from probe_replica_api import ReplicaAPI
from run_api_t0 import RateLimiter
import models_v51 as routing

ROOT = Path(__file__).resolve().parents[2]
CC = ROOT / 'obs/notes/traj_replica_20260929/tiered_binding_20261002/bin/cc'


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=1))
    tmp.replace(path)


class BudgetStop(RuntimeError):
    pass


class Ledger:
    """Durable reservation BEFORE dispatch, shared across threads/processes/restarts."""
    def __init__(self, root, caps):
        self.root, self.caps = Path(root), caps
        self.root.mkdir(parents=True, exist_ok=True)

    def rows(self):
        p = self.root / 'dispatch.jsonl'
        return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []

    def reserve(self, model, key, label, request_limit=2):
        with (self.root / 'dispatch.lock').open('a') as guard:
            fcntl.flock(guard, fcntl.LOCK_EX)
            rows = self.rows()
            used = [r for r in rows if r['model'] == model]
            if (self.root / ('STOP_' + model)).exists():
                raise BudgetStop('Persistent quota/auth stop: ' + model)
            if len(used) >= self.caps.get(model, 0):
                raise BudgetStop('Dispatch cap: ' + model)
            attempts = sum(r['request_sha256'] == key for r in used)
            if attempts >= request_limit:
                raise BudgetStop('Logical request retry cap: ' + label)
            row = dict(model=model, request_sha256=key, label=label,
                       attempt=attempts + 1, dispatch_index=len(rows) + 1, time=time.time())
            with (self.root / 'dispatch.jsonl').open('a') as f:
                f.write(json.dumps(row) + '\n'); f.flush(); os.fsync(f.fileno())
            return row

    def stop(self, model, reason):
        write_json(self.root / ('STOP_' + model), dict(reason=reason, time=time.time()))


class BoundedAPI(ReplicaAPI):
    def __init__(self, out, model, ledger, effort, limiter=None, max_tokens=8192):
        if model.startswith(('claude', 'gemini')):
            raise ValueError('Claude requires cc; Gemini requires agy. API route prohibited.')
        super().__init__(out, model, rpm=30, max_tokens=max_tokens)
        ep = routing.tm.MODELS[model]
        self.base = ep.base
        self.key = dotenv_values(ep.key_file).get(ep.key_var)
        if not self.key:
            raise ValueError('Missing configured credential for ' + model)
        self.endpoint = ep
        self.ledger, self.effort = ledger, effort
        self.limiter = limiter or RateLimiter(30)

    def call(self, label, msgs, max_tokens=None, *, replicate_id, temperature=.7, top_p=.9):
        safe, images = self.safe_messages(msgs)
        ep = self.endpoint
        payload = dict(model=self.model, messages=msgs,
                       temperature=ep.fixed_temperature if ep.fixed_temperature is not None else temperature,
                       top_p=ep.fixed_top_p if ep.fixed_top_p is not None else top_p,
                       max_tokens=max_tokens or self.max_tokens)
        if self.effort:
            payload['reasoning_effort'] = self.effort
        signature = dict(payload, messages=safe, replicate_id=replicate_id)
        key = digest(signature)
        dest = self.out / 'calls' / (key + '.json')
        with (self.out / 'locks' / (key + '.lock')).open('a') as guard:
            fcntl.flock(guard, fcntl.LOCK_EX)
            if dest.exists():  # Failed results are frozen too. No reset-to-zero retry on resume.
                return json.loads(dest.read_text())
            start, errors, result = time.monotonic(), [], None
            for _ in range(2):
                self.limiter.wait()
                dispatch = self.ledger.reserve(self.model, key, label)
                attempt = dispatch['attempt']
                try:
                    response = requests.post(self.base + '/v1/chat/completions', json=payload,
                                             headers={'Authorization': 'Bearer ' + self.key}, timeout=(20, 300))
                    if response.status_code != 200:
                        errors.append(dict(status=response.status_code, attempt=attempt,
                                           detail=response.text.replace(self.key, '[REDACTED]')[:600]))
                        if response.status_code in (401, 402, 403, 429):
                            self.ledger.stop(self.model, 'HTTP ' + str(response.status_code))
                        if response.status_code < 500:
                            break
                    else:
                        data = response.json(); choice = data['choices'][0]; m = choice['message']
                        result = dict(content=m.get('content') or '', reasoning_content=m.get('reasoning_content') or '',
                                      finish_reason=choice.get('finish_reason'), returned_model=data.get('model'),
                                      usage=data.get('usage', {}), response_id=data.get('id'))
                        if not result['returned_model'] or not result['returned_model'].startswith(self.model):
                            result['finish_reason'] = 'error'
                            errors.append(dict(error_type='returned_model_mismatch', attempt=attempt))
                        break
                except (requests.RequestException, ValueError, KeyError, IndexError) as exc:
                    errors.append(dict(error_type=type(exc).__name__, attempt=attempt))
                if attempt >= 2:
                    break
                time.sleep(2)
            result = result or dict(content='', reasoning_content='', finish_reason='error', returned_model=None, usage={})
            result.update(label=label, request=signature, request_sha256=key, images=images,
                          route=self.base, errors=errors, elapsed_s=round(time.monotonic() - start, 3))
            write_json(dest, result)
            return result


def parse_cc(stdout, staged, requested_model):
    events = []
    for line in stdout.splitlines():
        try:
            e = json.loads(line)
            if isinstance(e, dict): events.append(e)
        except ValueError:
            pass
    calls, successful, models, result = {}, set(), set(), None
    unexpected = []
    for e in events:
        if e.get('type') == 'system' and e.get('subtype') == 'init' and e.get('model'):
            models.add(e['model'])
        msg = e.get('message') if isinstance(e.get('message'), dict) else {}  # some events carry a plain string
        if e.get('type') == 'assistant' and msg.get('model'):
            models.add(msg['model'])
        for b in msg.get('content', []) if isinstance(msg.get('content'), list) else []:
            if b.get('type') == 'tool_use':
                name, path = b.get('name'), b.get('input', {}).get('file_path', '')
                calls[b.get('id')] = (name, path)
                if name != 'Read': unexpected.append(name)
            if b.get('type') == 'tool_result' and not b.get('is_error'):
                content = b.get('content')
                if isinstance(content, list) and any(c.get('type') == 'image' for c in content if isinstance(c, dict)):
                    successful.add(b.get('tool_use_id'))
        if e.get('type') == 'result': result = e
    wanted = {str(p.resolve()) for p in staged}
    opened = set()
    for tid in successful:
        name, path = calls.get(tid, (None, ''))
        if name == 'Read' and path:
            p = Path(path)
            resolved = str((staged[0].parent / p).resolve()) if not p.is_absolute() and staged else str(p.resolve())
            opened.add(resolved)
    bad_paths = []
    for name, path in calls.values():
        if name == 'Read':
            p = Path(path)
            resolved = str((staged[0].parent / p).resolve()) if not p.is_absolute() and staged else str(p.resolve())
            if resolved not in wanted: bad_paths.append(path)
    meta = dict(returned_models=sorted(models), successful_image_reads=sorted(opened),
                all_images_opened=wanted <= opened, tools_used=list(calls.values()),
                disallowed_tools=unexpected, unexpected_paths=bad_paths,
                text=(result or {}).get('result', ''), num_turns=(result or {}).get('num_turns'),
                usage=(result or {}).get('usage'), model_usage=(result or {}).get('modelUsage'),
                cost_usd=(result or {}).get('total_cost_usd'), is_error=(result or {}).get('is_error'))
    meta['valid'] = bool(result and not result.get('is_error') and meta['text'] and
                         models and all(re.fullmatch(re.escape(requested_model) + r'(?:-\d{8})?', m) for m in models)
                         and wanted <= opened and not unexpected and not bad_paths)
    return meta


class CCReviewer:
    """One independent official cc process per role. Only neutral staged images are accessible."""
    model = 'claude-sonnet-5-5'

    def __init__(self, out, ledger, model='claude-sonnet-5-5'):
        if model not in ('claude-sonnet-5-5', 'claude-opus-5-5'):
            raise ValueError('Unsupported cc model')
        self.out, self.ledger, self.model = Path(out), ledger, model
        self.out.mkdir(parents=True, exist_ok=True)

    def call(self, label, system, prompt, views):
        image_meta = [dict(view_id=v['view_id'], sha256=hashlib.sha256(Path(v['path']).read_bytes()).hexdigest()) for v in views]
        key = digest(dict(model=self.model, system=system, prompt=prompt, images=image_meta, label=label))
        dest = self.out / (key + '.json')
        if dest.exists(): return json.loads(dest.read_text())
        wd = Path('${WORK_DIR}') / ('tiered_' + uuid.uuid4().hex)
        wd.mkdir(parents=True)
        staged = []
        for i, v in enumerate(views):
            target = wd / ('obs' + str(i) + Path(v['path']).suffix)
            shutil.copyfile(v['path'], target); staged.append(target)
        mapping = '\n'.join(f"[{v['view_id']}] = {p.name}" for v, p in zip(views, staged))
        actual_prompt = ('Read every listed image file using Read. Do not read any other file.\n' + mapping + '\n\n' + prompt
                         if views else prompt)
        cmd = ['bash', str(CC), '-p', '--restricted', '--strict-mcp-config', '--disable-slash-commands',
               '--setting-sources', '', '--model', self.model, '--effort', 'medium',
               '--tools', 'Read' if views else '', '--allowedTools', 'Read',
               '--no-session-persistence', '--output-format', 'stream-json', '--verbose',
               '--system-prompt', system, actual_prompt]
        self.ledger.reserve(self.model, key, label, request_limit=2)  # one retry after an interrupted call
        write_json(self.out / (key + '.request.json'), dict(command=cmd, cwd=str(wd), images=image_meta))
        start = time.monotonic()
        try:
            p = subprocess.run(cmd, cwd=wd, capture_output=True, text=True, timeout=300, stdin=subprocess.DEVNULL)
            stdout, stderr, code = p.stdout, p.stderr, p.returncode
        except subprocess.TimeoutExpired as exc:
            def string(x): return x.decode(errors='replace') if isinstance(x, bytes) else x or ''
            stdout, stderr, code = string(exc.stdout), string(exc.stderr) + '\nTIMEOUT', 'timeout'
        (self.out / (key + '.stdout.jsonl')).write_text(stdout)
        (self.out / (key + '.stderr.txt')).write_text(stderr)
        receipt = parse_cc(stdout, staged, self.model)
        receipt.update(exit=code, request_sha256=key, model=self.model, route='bash task-local cc -> official claude',
                       label=label, images=image_meta, elapsed_s=round(time.monotonic() - start, 3))
        receipt['valid'] = receipt['valid'] and code == 0
        write_json(dest, receipt)
        return receipt
