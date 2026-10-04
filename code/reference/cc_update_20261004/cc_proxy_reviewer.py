"""Reading snapshot of the cc Sonnet adapter. Local paths and account details omitted.
Set explicit environment paths only in a separately authorized deployment.
Dependencies and operational configuration are not part of this publication.
"""
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
import uuid
from pathlib import Path

from probe_api import digest
from tiered_transport import parse_cc, write_json

CC_BIN = os.environ['TRAJECTORY_CC_BIN']
# claude-kimi defaults CLAUDE_CODE_EFFORT_LEVEL=max, which overrides --effort (smoke: 37k thinking tokens, 300 s timeout)
ENV = dict(os.environ, CLAUDE_CODE_EFFORT_LEVEL='medium')
SLOT = {'claude-sonnet-5-5': 'haiku', 'claude-opus-5-5': 'fable'}


class CCProxyReviewer:
    route = 'cc wrapper -> existing proxy chain -> Claude Sonnet'

    def __init__(self, out, ledger, model='claude-sonnet-5-5'):
        if model not in SLOT:
            raise ValueError('Unsupported cc model')
        self.out, self.ledger, self.model = Path(out), ledger, model
        self.ledger_model = 'cc-' + model
        self.out.mkdir(parents=True, exist_ok=True)

    def call(self, label, system, prompt, views):
        image_meta = [dict(view_id=v['view_id'], sha256=hashlib.sha256(Path(v['path']).read_bytes()).hexdigest()) for v in views]
        key = digest(dict(model=self.model, system=system, prompt=prompt, images=image_meta, label=label, route='cc'))
        dest = self.out / (key + '.json')
        if dest.exists():
            return json.loads(dest.read_text())
        wd = Path(os.environ['TRAJECTORY_CLI_WORKDIR']) / ('ccproxy_' + uuid.uuid4().hex)
        wd.mkdir(parents=True)
        staged = []
        for i, v in enumerate(views):
            target = wd / ('obs' + str(i) + Path(v['path']).suffix)
            shutil.copyfile(v['path'], target)
            staged.append(target)
        mapping = '\n'.join(f"[{v['view_id']}] = {p.name}" for v, p in zip(views, staged))
        actual_prompt = ('Read every listed image file using Read. Do not read any other file.\n' + mapping + '\n\n' + prompt
                         if views else prompt)
        cmd = [CC_BIN, '-p', '--permission-mode', 'default', '--strict-mcp-config', '--disable-slash-commands',
               '--setting-sources', '', '--model', SLOT[self.model], '--effort', 'medium',
               '--tools', 'Read' if views else '', '--allowedTools', 'Read',
               '--no-session-persistence', '--output-format', 'stream-json', '--verbose',
               '--system-prompt', system, actual_prompt]
        self.ledger.reserve(self.ledger_model, key, label, request_limit=2)
        write_json(self.out / (key + '.request.json'), dict(command=cmd, cwd=str(wd), images=image_meta))
        start = time.monotonic()
        try:
            p = subprocess.run(cmd, cwd=wd, capture_output=True, text=True, timeout=300, stdin=subprocess.DEVNULL,
                               env=ENV)
            stdout, stderr, code = p.stdout, p.stderr, p.returncode
        except subprocess.TimeoutExpired as exc:
            def string(x): return x.decode(errors='replace') if isinstance(x, bytes) else x or ''
            stdout, stderr, code = string(exc.stdout), string(exc.stderr) + '\nTIMEOUT', 'timeout'
        (self.out / (key + '.stdout.jsonl')).write_text(stdout)
        (self.out / (key + '.stderr.txt')).write_text(stderr)
        receipt = parse_cc(stdout, staged, self.model)
        models = receipt['returned_models']
        same_model = bool(models) and all(re.fullmatch(re.escape(self.model) + r'(?:-\d{8})?(?:\[1m\])?', m) for m in models)
        receipt['valid'] = bool(code == 0 and same_model and receipt['text'] and not receipt['is_error']
                                and receipt['all_images_opened'] and not receipt['disallowed_tools']
                                and not receipt['unexpected_paths'])
        receipt.update(exit=code, request_sha256=key, model=self.model, slot=SLOT[self.model], route=self.route,
                       label=label, images=image_meta, elapsed_s=round(time.monotonic() - start, 3))
        write_json(dest, receipt)
        return receipt
