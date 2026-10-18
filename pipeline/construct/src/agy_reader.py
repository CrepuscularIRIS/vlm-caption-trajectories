"""Gemini blind reader through the official agy CLI (Gemini only via agy).

Same interface as tiered_transport.CCReviewer: call(label, system, prompt, views) -> receipt dict with
'valid', 'text', 'request_sha256'. Each call is a fresh agy process in a fresh directory that holds ONLY the
staged view images (copied under neutral names). agy has no system-prompt flag, so the system text is put first.
A receipt is valid only if: result status SUCCESS, non-empty text, the init model is the requested model, every
staged image was opened with view_file and no other path or tool (besides view_file / finish) was used.
"""
import hashlib
import json
import shutil
import subprocess
import time
import uuid
from pathlib import Path

from probe_api import digest
from tiered_transport import write_json

AGY = '${AGY_BIN}'
WORK = Path('${WORK_DIR}')
ALLOWED_TOOLS = {'view_file', 'finish'}


def parse_agy(stdout: str, staged: list[Path], requested_model: str) -> dict:
    events = []
    for line in stdout.splitlines():
        try:
            e = json.loads(line)
            if isinstance(e, dict):
                events.append(e)
        except ValueError:
            pass
    model, result, tools, opened, bad_paths = None, None, [], set(), []
    wanted = {str(p.resolve()) for p in staged}
    for e in events:
        if e.get('event') == 'init':
            model = (e.get('init') or {}).get('model')
        su = e.get('step_update') or {}
        if su.get('step_type') == 'tool' and su.get('state') == 'DONE':
            name = su.get('tool_name')
            params = (su.get('tool_info') or {}).get('parameters') or {}
            path = params.get('AbsolutePath') or params.get('path') or ''
            tools.append((name, path))
            if name == 'view_file':
                rp = str(Path(path).resolve()) if path else ''
                if rp in wanted:
                    opened.add(rp)
                else:
                    bad_paths.append(path)
        if e.get('event') == 'result':
            result = e.get('result') or {}
    text = (result or {}).get('response') or ''
    disallowed = sorted({n for n, _ in tools if n not in ALLOWED_TOOLS})
    meta = dict(returned_models=[model] if model else [], text=text, status=(result or {}).get('status'),
                tools_used=tools, successful_image_reads=sorted(opened), all_images_opened=wanted <= opened,
                disallowed_tools=disallowed, unexpected_paths=bad_paths,
                duration_s=(result or {}).get('duration_seconds'))
    meta['valid'] = bool(result and meta['status'] == 'SUCCESS' and text.strip() and model == requested_model
                         and wanted <= opened and not disallowed and not bad_paths)
    return meta


class AgyReader:
    def __init__(self, out, ledger, model='gemini-3.8-flash-low'):
        self.out, self.ledger, self.model = Path(out), ledger, model
        self.out.mkdir(parents=True, exist_ok=True)

    def call(self, label, system, prompt, views):
        image_meta = [dict(view_id=v['view_id'], sha256=hashlib.sha256(Path(v['path']).read_bytes()).hexdigest())
                      for v in views]
        key = digest(dict(model=self.model, system=system, prompt=prompt, images=image_meta, label=label, route='agy'))
        dest = self.out / (key + '.json')
        if dest.exists():
            cached = json.loads(dest.read_text())
            if cached.get('valid') or label.endswith('/retry'):
                return cached
            return self.call(label + '/retry', system, prompt + '\nAnswer directly after opening the images. '
                             'Never run commands or use any tool other than view_file.', views)
        wd = WORK / ('agy_' + uuid.uuid4().hex)
        wd.mkdir(parents=True)
        staged = []
        for i, v in enumerate(views):
            target = wd / ('obs' + str(i) + Path(v['path']).suffix)
            shutil.copyfile(v['path'], target)
            staged.append(target)
        mapping = '\n'.join(f"[{v['view_id']}] = {p.name}" for v, p in zip(views, staged))
        full = (system + '\n\n' + 'Open every listed image file in the current directory with view_file. '
                'Do not open any other file, run commands or browse.\n' + mapping + '\n\n' + prompt)
        cmd = [AGY, '-p', full, '--model', self.model, '--output-format', 'stream-json', '--sandbox',
               '--disable-slash-commands']
        self.ledger.reserve(self.model, key, label, request_limit=2)  # one retry after an interrupted call
        write_json(self.out / (key + '.request.json'), dict(command=cmd[:1] + ['-p', '<prompt>'] + cmd[3:], prompt=full,
                                                             cwd=str(wd), images=image_meta))
        start = time.monotonic()
        try:
            p = subprocess.run(cmd, cwd=wd, capture_output=True, text=True, timeout=300, stdin=subprocess.DEVNULL)
            stdout, stderr, code = p.stdout, p.stderr, p.returncode
        except subprocess.TimeoutExpired as exc:
            def s(x): return x.decode(errors='replace') if isinstance(x, bytes) else x or ''
            stdout, stderr, code = s(exc.stdout), s(exc.stderr) + '\nTIMEOUT', 'timeout'
        (self.out / (key + '.stdout.jsonl')).write_text(stdout)
        (self.out / (key + '.stderr.txt')).write_text(stderr)
        receipt = parse_agy(stdout, staged, self.model)
        receipt.update(exit=code, request_sha256=key, model=self.model, route='agy official CLI', label=label,
                       images=image_meta, elapsed_s=round(time.monotonic() - start, 3))
        receipt['valid'] = receipt['valid'] and code == 0
        write_json(dest, receipt)
        if not receipt['valid'] and not label.endswith('/retry'):
            # Gemini sometimes tries to run a command after viewing (auto-denied) and then returns no text: one retry.
            return self.call(label + '/retry', system, prompt + '\nAnswer directly after opening the images. '
                             'Never run commands or use any tool other than view_file.', views)
        return receipt
