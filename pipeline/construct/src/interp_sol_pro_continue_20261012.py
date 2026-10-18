"""Native Pro-only Sol continuation authorized by the user to continue without another approval pause.

Keeps all previously issued sources, episodes and STOP files unchanged.
"""
import argparse
import copy
import base64
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import tempfile
import time

from dotenv import dotenv_values
import requests
import yaml

import interp_sol_full_campaign as campaign
import interp_sol_full_4040 as batch
from tiered_transport import ROOT, write_json

PORT = 8322
ENDPOINT = f'http://127.0.0.1:{PORT}'
MODEL = 'gpt-5.6-sol'
ALLOWED_RETURNED_MODELS = ['gpt-5.6-sol']
WORKERS, RPM = 48, 90
REAUTH_AFTER = datetime.fromisoformat('2026-10-12T06:50:00+00:00').timestamp()
PROXY_RECORD = 'SOL_RESUMED_PRO_PROXY_LAUNCH.json'
AUTHORIZATION = '2026-10-12 latest user: do not use Grok; use GPT5.6Sol to finish all remaining tasks quickly; do not pause again. Pro-only Sol48/90RPM low, real observations and local intervention; preserve previous episodes; bounded cooldown continuation on never-issued tasks for known transient failures.'


def credential_identity(data):
    token = data['id_token'].split('.')[1]
    claims = json.loads(base64.urlsafe_b64decode(token + '=' * (-len(token) % 4)))
    auth = claims['https://api.openai.com/auth']
    user = auth.get('chatgpt_user_id') or claims.get('sub')
    if not user:
        raise ValueError('Missing credential user identity')
    return auth.get('chatgpt_plan_type'), hashlib.sha256(user.encode()).hexdigest()[:12]


def select_recent_auth(source):
    selected = []
    for path in source.glob('*.json'):
        data = batch.read(path)
        if data.get('type') == 'xai' and not data.get('disabled', False):
            selected.append(path)
    if len(selected) != 1:
        raise ValueError('Expected exactly one existing active native xAI credential')
    return selected


def assert_idle(parent):
    if (parent/'PAUSE').exists():
        raise ValueError('Campaign PAUSE')
    for loc in campaign.locations(parent):
        for role in batch.ROLES:
            out = loc/role
            if (out/'LAUNCH.json').exists() and not (out/'RUN_STATE.json').exists():
                raise ValueError('Unconfirmed launch; inspect before dispatch')
            if (out/'RUN_STATE.json').exists() and batch.read(out/'RUN_STATE.json')['state'] == 'running':
                raise ValueError('Another wave running; wait for settlement')
    return campaign.inventory(parent)


def set_route():
    tm = batch.old.transport.base.routing.tm
    tm.MODELS[MODEL] = replace(tm.MODELS[MODEL], base=ENDPOINT,
                              key_file=tm.REPO_ENV, key_var='LITELLM_API_KEY')



PROMPT_CONDITION = 'observe_first_real_crop_pro_20261012'
OBSERVE_FIRST = """Trajectory construction condition: begin with a real observation action.
On the first assistant turn, locate evidence relevant to this question and request ONE real crop of that region. Do not emit the final answer on this first turn, even when the full image seems sufficient.
Use the existing short Caption-plus-one-action format. State one to three short facts actually visible in [original_image] and why the chosen crop will inspect or confirm relevant evidence. Do not invent uncertainty or errors; confirmation is a valid purpose when evidence already seems sufficient. Do not put a final-answer conclusion in the first caption.
The crop must be question-relevant and refer to an actually available view. Subsequent captions and actions must use actual returned pixels and preserve reliable earlier facts. There is no fixed five-step requirement: after observing the returned crop and any local intervention, answer when the available evidence supports stopping. Do not fabricate observations, intermediate results, or hidden thought processes."""
CONTINUE_OBSERVATIONS = """Continue from the actual observations and reliable earlier facts. Describe briefly what the latest crop confirms or misses; do not pretend a gap exists when earlier evidence already suffices. Choose another real action only for a concrete unresolved detail. Answer once evidence is sufficient; there is no fixed target length."""


def augment_messages(label, messages):
    result = copy.deepcopy(messages)
    if label.endswith('/editor'):
        return result
    extra = OBSERVE_FIRST if label.endswith('/fresh_s1') else CONTINUE_OBSERVATIONS
    for message in result:
        if message.get('role') != 'system':
            continue
        content = message.get('content')
        if isinstance(content, str):
            message['content'] += '\n\n' + extra
            return result
        if isinstance(content, list) and all(block.get('type') == 'text' for block in content):
            content.append({'type':'text', 'text':extra})
            return result
    raise ValueError('Expected a text-only system protocol')


class ObserveFirstTransport(batch.Transport):
    def call(self, label, msgs, max_tokens=None, *, replicate_id, temperature=.7, top_p=.9):
        # Add before the frozen transport hashes/persists the request, so receipts
        # describe exactly the messages actually sent, including this condition.
        messages = augment_messages(label, msgs)
        return super().call(label, messages, max_tokens=max_tokens, replicate_id=replicate_id,
                            temperature=temperature, top_p=top_p)


ORIGINAL_GENERATE = batch.generate


def generate_observation_condition(parent, packet, api, private):
    result = ORIGINAL_GENERATE(parent, packet, api, private)
    result['prompt_condition'] = PROMPT_CONDITION
    result['prompt_condition_sha256'] = hashlib.sha256(OBSERVE_FIRST.encode()).hexdigest()
    export = result['rollout'].get('export')
    if export:
        system = export['conversations'][0]
        if system['from'] != 'system':
            raise ValueError('Unexpected export protocol')
        system['value'] += '\n\n' + OBSERVE_FIRST
        export['generation_condition'] = PROMPT_CONDITION
    write_json(parent/api.role/api.role/'results'/(packet['audit_id']+'.json'), result)
    return result


def install_run():
    set_route()
    batch.Transport = ObserveFirstTransport
    batch.generate = generate_observation_condition



def prepare_primary_config(parent, loc):
    if any((loc/'sol'/n).exists() for n in ('LAUNCH.json','RUN_STATE.json','PAUSE')):
        raise ValueError('Never edit an issued wave')
    issued, _ = campaign.inventory(parent)
    proxy = batch.read(parent/PROXY_RECORD)
    upcfg = Path(proxy['runtime_dir'])/'cliproxy.yaml'
    if batch.old.sha(upcfg) != proxy['config_sha256']:
        raise ValueError('Owned proxy changed')
    cfg=batch.read(loc/'config.json')
    cfg['models']['sol'].update(model=MODEL,endpoint=ENDPOINT,workers=WORKERS,rpm=RPM,effort='low',allowed_returned_models=ALLOWED_RETURNED_MODELS)
    cfg['actual_primary_model']=MODEL
    cfg['construction_prompt']=dict(condition=PROMPT_CONDITION,first_turn=OBSERVE_FIRST,continuation=CONTINUE_OBSERVATIONS,semantic_verified=False)
    cfg['provider_route']='native Pro-only Codex GPT5.6Sol; no Cursor/xAI'
    cfg['frozen_files'][str(Path(__file__).resolve())]=batch.old.sha(__file__)
    cfg['frozen_files'][str(upcfg)]=batch.old.sha(upcfg)
    write_json(loc/'config.json',cfg);(loc/'CONFIG.sha256').write_text(batch.old.sha(loc/'config.json')+'\n')
    branch=dict(cfg,frozen_files=dict(cfg['frozen_files'],**{str(loc/'config.json'):batch.old.sha(loc/'config.json')}))
    write_json(loc/'sol/config.json',branch);(loc/'sol/CONFIG.sha256').write_text(batch.old.sha(loc/'sol/config.json')+'\n')
    batch.verify(loc,'sol');ids={row['audit_id'] for row in batch.read(loc/'SCOPE_ALL_4040.json')}
    if ids & issued['sol'] or len(ids)!=4040-len(issued['sol']):raise ValueError('Unexpected unissued primary scope')
    return ids


def configure(parent):
    batch.verify(parent)
    issued, _ = assert_idle(parent)
    old = parent/'continuations/sol_008'
    out = old/'sol'
    if batch.read(out/'RUN_STATE.json')['state'] != 'stopped_incomplete':
        raise ValueError('Unexpected previous Grok state')
    calls = [batch.read(f) for f in (out/'sol/calls').glob('*.json')]
    if len(calls) != 3225 or not any(not c['errors'] for c in calls):
        raise ValueError('Expected fully inspected prior Grok wave')
    for c in calls:
        for error in c['errors']:
            if error.get('status') != 500 or not any(x in str(error.get('detail', {})) for x in ('EOF', 'lookup cli-chat-proxy.grok.com')):
                raise ValueError('New quota/auth/identity failure is outside this resumption')
    if batch.pending_requests(old,'sol'):
        raise ValueError('Prior Grok requests not settled')
    proxy = batch.read(parent/PROXY_RECORD)
    for proc in proxy['processes']:
        observed=Path('/proc')/str(proc['pid'])/'cmdline'
        if not observed.exists() or observed.read_bytes().decode().strip('\0').split('\0') != proc['command']:
            raise ValueError('Owned Pro proxy process identity changed')
    private = Path(proxy['runtime_dir']);upcfg = private/'cliproxy.yaml'
    if batch.old.sha(upcfg) != proxy['config_sha256']:
        raise ValueError('Owned native proxy configuration changed')
    config = yaml.safe_load(upcfg.read_text())
    auths = [batch.read(f) for f in Path(config['auth-dir']).glob('*.json')]
    if len(auths) != 1 or auths[0].get('type') != 'codex' or credential_identity(auths[0])[0] != 'pro' or config['port'] != PORT or config['plugins']['enabled']:
        raise ValueError('Expected only the unchanged Pro-only Codex provider')
    tm = batch.old.transport.base.routing.tm;clientkey = dotenv_values(tm.REPO_ENV)['LITELLM_API_KEY']
    if config['api-keys'] != [clientkey]:raise ValueError('Private endpoint auth changed')
    reply = requests.get(ENDPOINT+'/v1/models',headers={'Authorization':'Bearer '+clientkey},timeout=10)
    reply.raise_for_status()
    if not any(m['id']==MODEL and m.get('owned_by')=='openai' for m in reply.json().get('data',[])):
        raise ValueError('Pro-only Sol metadata unavailable')
    if time.time() > batch.read(parent/'config.json')['deadline']:raise ValueError('Deadline elapsed')
    plan=batch.read(parent/campaign.REGISTRY)
    plan['blocked'].pop('sol',None)
    quota_wave=parent/'continuations/k3_005'
    if batch.read(quota_wave/'k3/RUN_STATE.json')['state'] != 'stopped_incomplete' or batch.pending_requests(quota_wave,'k3'):
        raise ValueError('K3 quota wave must be stopped and settled')
    plan['blocked'].pop('k3',None)
    plan['acknowledged_stopped_waves']=sorted(set(plan.get('acknowledged_stopped_waves',[]))|{str(quota_wave)})
    plan['not_before_utc']['k3']='2026-10-12T14:20:14.022480+00:00'
    plan['acknowledged_stopped_waves']=sorted(set(plan.get('acknowledged_stopped_waves',[]))|{str(old)})
    plan['not_before_utc']['sol']=datetime.now(timezone.utc).isoformat()
    plan['active_sol_route']=dict(endpoint=ENDPOINT,provider='codex',actual_model=MODEL,
        allowed_returned_models=ALLOWED_RETURNED_MODELS,credentials=1,workers=WORKERS,rpm=RPM,effort='low',
        entrypoint=str(Path(__file__).resolve()),authorization=AUTHORIZATION,
        accounting_slot='immutable legacy primary slot sol; actual teacher GPT5.6Sol Pro-only Codex')
    write_json(parent/campaign.REGISTRY,plan)
    loc=campaign.prepare_wave(parent,'sol')
    if any((loc/'sol'/n).exists() for n in ('LAUNCH.json','RUN_STATE.json','PAUSE')):raise ValueError('Fresh unissued wave required')
    ids=prepare_primary_config(parent,loc)
    record=dict(at=batch.now(),wave=str(loc),authorization=AUTHORIZATION,tasks=len(ids),workers=WORKERS,rpm=RPM,effort='low',endpoint=ENDPOINT,actual_model=MODEL,
                allowed_returned_models=ALLOWED_RETURNED_MODELS,provider='codex',old_STOP_and_sources_preserved=True,first_actual_receipt_pending=True,
                shared_services_modified=False,resumption_reason='Latest user authorizes continued execution without another approval pause; latest user switches back to Pro-only Sol48/90; no prior issued task reissued',
                primary_history_issued=len(issued['sol']),legacy_accounting_slot='sol')
    write_json(parent/'SOL_CONTINUE_AUTHORIZATION_20261012.json',record)
    write_json(parent/'SOL_CONTINUE_CONFIG_RECORD.json',record)
    return dict(wave=str(loc),tasks=len(ids),model=MODEL,allowed_returned_models=ALLOWED_RETURNED_MODELS,workers=WORKERS,rpm=RPM,effort='low',generation_calls=0)


def launch(parent, loc):
    assert_idle(parent)
    cfg = batch.verify(loc, 'sol')
    out = loc/'sol'
    if batch.read(parent/campaign.REGISTRY)['blocked'].get('sol'):
        raise ValueError('Sol requires a new user decision')
    if cfg['models']['sol']['endpoint'] != ENDPOINT or time.time() > cfg['deadline']:
        raise ValueError('Route/deadline mismatch')
    if any((out/n).exists() for n in ('LAUNCH.json', 'RUN_STATE.json', 'PAUSE')) or (loc/'PAUSE').exists():
        raise ValueError('Never relaunch or bypass PAUSE')
    command = ['${PYTHON}', '-u', str(Path(__file__).resolve()),
               'run', '--parent', str(parent), '--wave', str(loc)]
    with (out/'worker.log').open('ab') as log:
        proc = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True,
                                env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
    write_json(out/'LAUNCH.json', dict(pid=proc.pid, command=command, at=batch.now(), first_real_receipt_pending=True))
    return dict(launched=True, wave=str(loc), tasks=cfg['tasks'], workers=WORKERS, rpm=RPM,
                effort='low', first_real_receipt_pending=True)


def recoverable_errors(role, calls):
    errors=[e for c in calls for e in c.get('errors',[])]
    if not errors:
        return None
    if role=='k3' and all(e.get('status')==500 and '5-hour usage limit' in str(e.get('detail',{})) for e in errors):
        return 'quota_5h'
    if role=='sol' and all(e.get('status') in (500,502,503) and any(token in str(e.get('detail',{})) for token in ('EOF','dial tcp: lookup','server_is_overloaded','TLS handshake','http2:')) for e in errors):
        return 'transient_connection'
    return None


def recover_stopped(parent, plan):
    acknowledged=set(plan.get('acknowledged_stopped_waves',[]))
    counters=plan.setdefault('authorized_recovery_counts',{})
    for wave in plan['waves']:
        if wave['parent'] in acknowledged:
            continue
        loc=Path(wave['parent']);role=wave['role'];out=loc/role
        if not (out/'RUN_STATE.json').exists():
            continue
        state=batch.read(out/'RUN_STATE.json')
        if state['state']!='stopped_incomplete':
            continue
        if (loc/'PAUSE').exists() or (out/'PAUSE').exists() or batch.pending_requests(loc,role):
            continue
        cfg=batch.verify(loc,role)
        calls=[batch.read(f) for f in (out/role/'calls').glob('*.json')]
        attempts=[batch.read(f) for f in (out/role/'attempts').glob('*.json')]
        if len(attempts)!=len(calls) or any(a['state']!='settled' or batch.old.sha(Path(a['receipt']))!=a['receipt_sha256'] for a in attempts):
            continue
        allowed=cfg['models'][role].get('allowed_returned_models',[cfg['models'][role]['model']])
        if any(c.get('config_sha256')!=batch.old.sha(out/'config.json') or (not c.get('errors') and c.get('returned_model') not in allowed) for c in calls):
            continue
        cause=recoverable_errors(role,calls)
        if cause is None or counters.get(role,0)>=3:
            continue
        waits=5*3600 if cause=='quota_5h' else 30*60
        gate=datetime.fromisoformat(state['finished_at']).timestamp()+waits
        plan['not_before_utc'][role]=datetime.fromtimestamp(max(gate,datetime.fromisoformat(plan['not_before_utc'][role]).timestamp()),timezone.utc).isoformat()
        acknowledged.add(str(loc));plan['blocked'].pop(role,None)
        counters[role]=counters.get(role,0)+1
        plan.setdefault('automatic_recovery_records',[]).append(dict(wave=str(loc),role=role,cause=cause,not_before_utc=plan['not_before_utc'][role],authorization=AUTHORIZATION,old_STOP_preserved=True,prior_issued_tasks_not_retried=True))
    plan['acknowledged_stopped_waves']=sorted(acknowledged)
    write_json(parent/campaign.REGISTRY,plan)
    return plan


def tick(parent, do_launch=False):
    batch.verify(parent)
    plan = batch.read(parent/campaign.REGISTRY)
    if (parent/'PAUSE').exists():
        return dict(state='paused')
    for loc in campaign.locations(parent):
        for role in batch.ROLES:
            out = loc/role
            if (out/'LAUNCH.json').exists() and not (out/'RUN_STATE.json').exists():
                return dict(state='launch_unconfirmed', role=role, wave=str(loc))
            if (out/'RUN_STATE.json').exists() and batch.read(out/'RUN_STATE.json')['state'] == 'running':
                return dict(state='waiting_active_wave_settlement', role=role, wave=str(loc))
    plan=recover_stopped(parent,plan)
    events = []
    acknowledged = set(plan.get('acknowledged_stopped_waves', []))
    for role in batch.ROLES:
        if not plan['blocked'].get(role):
            stopped = [w for w in plan['waves'] if w['role'] == role and w['parent'] not in acknowledged
                       and (Path(w['parent'])/role/'RUN_STATE.json').exists()
                       and batch.read(Path(w['parent'])/role/'RUN_STATE.json')['state'] == 'stopped_incomplete']
            if stopped:
                plan['blocked'][role] = 'New continuation stopped; preserve receipts and inspect the new cause before recovery'
                write_json(parent/campaign.REGISTRY, plan)
        if plan['blocked'].get(role):
            events.append(dict(role=role, state='blocked', reason=plan['blocked'][role]))
            continue
        if time.time() > batch.read(parent/'config.json')['deadline']:
            events.append(dict(role=role, state='deadline_elapsed'))
            continue
        if time.time() < datetime.fromisoformat(plan['not_before_utc'][role]).timestamp():
            events.append(dict(role=role, state='waiting_window'))
            continue
        if not campaign.eligible_ids(parent, role):
            events.append(dict(role=role, state='no_confirmed_work'))
            continue
        loc = campaign.prepare_wave(parent, role)
        if role == 'sol' and batch.read(loc/'config.json')['models']['sol']['endpoint'] != ENDPOINT:
            prepare_primary_config(parent,loc)
        if do_launch:
            events.append(launch(parent, loc) if role == 'sol' else campaign.launch_wave(parent, loc, role))
            return dict(events=events)
        events.append(dict(role=role, state='ready', wave=str(loc)))
    return dict(events=events)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['configure', 'launch', 'run', 'tick'])
    parser.add_argument('--parent', type=Path, default=campaign.PARENT)
    parser.add_argument('--wave', type=Path)
    parser.add_argument('--launch', action='store_true')
    args = parser.parse_args()
    parent = args.parent.resolve()
    if args.action == 'configure':
        print(json.dumps(configure(parent)))
    elif args.action == 'launch':
        print(json.dumps(launch(parent, args.wave.resolve())))
    elif args.action == 'tick':
        print(json.dumps(tick(parent, args.launch)))
    else:
        install_run()
        campaign.run_wave(parent, args.wave.resolve(), 'sol')
