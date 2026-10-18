"""Prefix-only recovery calibration. Never replays the old successful suffix.

One editor-forced real crop, then ordinary v5.1 model continuation. Original
baseline and labels are read-only; the editor turn is context, never a target.
Gemini uses AGY, K3 the existing LiteLLM route; durable no-retry transport.
"""
import argparse
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading
import time

from PIL import Image

import caption_v51 as proto
from corpus_scoring import score
from export_protocol_check import check_turn
from monitor_step import expand_min_crop
from probe_api import digest, text_msg
from run_caption_v51 import view_record
from run_minio3_replica import resolve_source, save_obs
import sft_v2_topup as transport
from tiered_transport import BudgetStop, Ledger, ROOT, write_json

BASELINE = ROOT / 'obs/benchmarks/sft_v2_2_keep_20261009/MANIFEST.jsonl'
VERSION = 'interp-recovery-calibration-20261010'


class RecoveryTransport(transport.Transport):
    def check(self):
        if sha(self.batch/'config.json') != (self.batch/'CONFIG.sha256').read_text().strip():
            write_json(self.batch/'PAUSE',dict(reason='config_changed',time=time.time()))
            raise BudgetStop('Configuration changed')
        super().check()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def iou(a, b):
    inter = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter / union if union else 0


def perturbation(prefix_views, kind, seed):
    """Uses only the first received crop; geometry proposes, never certifies a miss."""
    if [v['view_id'] for v in prefix_views] != ['original_image', 'observation_1']:
        raise ValueError('Expected exactly the visible prefix')
    if kind == 'tight_context':
        return dict(source='observation_1', bbox=[.25, .25, .75, .75],
                    intent='tight framing; actual context loss requires visual review')
    if kind != 'shift_region':
        raise ValueError('Unknown intervention')
    f = prefix_views[1]['fov_original']
    w, h = min(.28, max(.12, f[2]-f[0])), min(.28, max(.12, f[3]-f[1]))
    boxes = [[round(x*(1-w), 5), round(y*(1-h), 5),
              round(x*(1-w)+w, 5), round(y*(1-h)+h, 5)]
             for x in (0, .5, 1) for y in (0, .5, 1)]
    ordered = sorted(boxes, key=lambda b: (iou(b, f), digest([seed, b])))
    return dict(source='original_image', bbox=ordered[0],
                intent='region shifted away from first received crop; not a semantic miss assertion',
                first_view_iou=round(iou(ordered[0], f), 5))


def choose(rows, allocation=None):
    allocation = allocation or {'Gemini': 28, 'K3': 12}
    eligible = [r for r in rows if r['model'] in allocation and 2 <= r['crops'] <= 7]
    selected, used_groups = [], set()
    for model, count in allocation.items():
        pools = defaultdict(list)
        for r in eligible:
            if r['model'] == model:
                pools[(r['bench'], r['material'])].append(r)
        for values in pools.values():
            values.sort(key=lambda r: digest([VERSION, r['canonical_sample_key']]))
        counts = Counter()
        for _ in range(count):
            active = [k for k, v in pools.items() if v]
            if not active:
                break  # never add another source just to fill a quota
            layer = min(active, key=lambda k: (counts[k], k))
            r = pools[layer].pop(0)
            if set(r['canonical_groups']) & used_groups:
                continue
            selected.append(r); counts[layer] += 1; used_groups.update(r['canonical_groups'])
    return selected


def public_prefix(result, manifest, audit_id, kind):
    """Construct public continuation input with an explicit field allowlist."""
    ro = result['rollout']
    c = ro['export']['conversations']
    if c[2]['from'] != 'gpt' or c[3]['from'] != 'human':
        raise ValueError('Unexpected initial transcript')
    p = proto.parse_step(c[2]['value'])
    if proto.check_step(p, 1, 0, ['original_image'], 8) or p['kind'] != 'grounding':
        raise ValueError('Original first turn is invalid')
    views = copy.deepcopy(ro['views'][:2])
    return dict(audit_id=audit_id, item_id=manifest['item_id'], question=result['item']['question'],
                image=result['item']['image'], source_image_sha256=sha(result['item']['image']),
                assigned_role=manifest['model'].lower(), intervention=kind,
                prefix_assistant=c[2]['value'], prefix_views=views,
                perturbation=perturbation(views, kind, audit_id), budget=8)


def prepare(out):
    if out.exists():
        raise ValueError('Fresh directory required')
    manifest = read_rows(BASELINE)
    if len(manifest) != 4040:
        raise ValueError('Baseline changed')
    selected = choose(manifest)
    out.mkdir(parents=True)
    packets, private = [], []
    model_counts = Counter()
    frozen = {str(BASELINE): sha(BASELINE)}
    for i, r in enumerate(selected, 1):
        path = Path(r['result_path'])
        if sha(path) != r['result_sha256']:
            raise ValueError('Source result changed')
        result = json.loads(path.read_text())
        role = r['model'].lower(); n = model_counts[role]; model_counts[role] += 1
        audit_id = f'C{i:03d}'
        packet = public_prefix(result, r, audit_id, 'shift_region' if n % 2 == 0 else 'tight_context')
        for v in packet['prefix_views']:
            if sha(v['path']) != v['sha256']:
                raise ValueError('Source prefix view changed')
        packet_path = out / 'inputs' / (audit_id+'.json')
        write_json(packet_path, packet); packets.append(packet)
        frozen[str(packet_path)] = sha(packet_path)
        private.append(dict(manifest=r, audit_id=audit_id, answer_spec=result['item']['answer_spec'],
                            source_effort=result.get('effort'), input_path=str(packet_path)))
    write_json(out/'SELECTION_PRIVATE.json', private)
    frozen[str(out/'SELECTION_PRIVATE.json')] = sha(out/'SELECTION_PRIVATE.json')
    files = ['interp_recovery_calibration.py', 'sft_v2_topup.py', 'agy_reader.py',
             *transport.base.CODE_FILES, 'export_protocol_check.py', 'caption_v3.py']
    frozen.update({str(ROOT/'obs/src'/p): sha(ROOT/'obs/src'/p) for p in set(files)})
    cfg = dict(version=VERSION, created_at=time.time(), deadline=time.time()+4*3600,
               authorization='2026-10-10 user: 开始执行, 可以用K3和Gemini去加入. Execute the previously proposed <=40-task calibration.',
               tasks=len(packets), workers=4, budget_crops=8, examples=[], frozen_files=frozen,
               baseline_T_Q_G=[4040,4040,3690], baseline_manifest=str(BASELINE),
               valid_answer_retries=0, technical_retries=0, format_retries=0,
               reviewer='Codex root actual-image review; not human acceptance',
               perturbations='one editor-forced shift or tight crop per task, chosen only from first received crop',
               continuation='same named generator family, fresh low-effort invocation; no old suffix/GT/audit label/demonstrations',
               masks='reliable prefix=1, editor-forced assistant body+EOS=0, approved continuation=1; must verify actual token labels',
               no_gpu=True, no_training=True, no_old_run_resume=True,
               budget_units='client HTTP for K3; AGY CLI invocations for Gemini, internal server retries unobserved',
               models={
                   'gemini':dict(model='gemini-3.8-flash-low', effort='low', route='agy', endpoint=transport.AGY,
                                 tasks=model_counts['gemini'],workers=2,rpm=12,call_cap=model_counts['gemini']*7,
                                 max_tokens=8192,allowed_returned_models=['gemini-3.8-flash-low']),
                   'k3':dict(model='k3-256k', effort='low', route='api', endpoint='http://127.0.0.1:4001',
                             tasks=model_counts['k3'],workers=2,rpm=12,call_cap=model_counts['k3']*7,
                             max_tokens=8192,allowed_returned_models=['k3-256k'])})
    write_json(out/'config.json', cfg)
    (out/'CONFIG.sha256').write_text(sha(out/'config.json')+'\n')
    write_json(out/'SELECTION_STATS.json',dict(tasks=len(packets),models=dict(model_counts),
               unique_tasks=len({r['canonical_sample_key'] for r in selected}),
               unique_groups=len({g for r in selected for g in r['canonical_groups']}),
               materials=dict(Counter(r['material'] for r in selected)),
               benchmarks=dict(Counter(r['bench'] for r in selected)),
               old_crops=dict(Counter(r['crops'] for r in selected)),
               interventions=dict(Counter(p['intervention'] for p in packets)),
               source_efforts=dict(Counter(str(r['source_effort']) for r in private))))
    brief = '''# 插值恢复校准工作包（唯一新 run）

用户授权执行首批至多40题：Gemini28、K3 12，现有基线同族多裁图路径、每原图组至多一题。
固定Database II为4040/4040/3690，不改旧导出、标签、GT和划分；无GPU、训练、Claude/Astra或其他模型。
原第一步Caption/实际crop不改，在第二步插一次真实工具扰动。只依据已经收到的第一幅crop选择
裁偏区域或紧框，不读取原后续框/Caption/答案来选择或续写。几何不证明无证据。
强插动作仅为历史上下文，body及EOS不监督；恢复全部由模型从可见前缀重新生成至终答，不回放旧后缀。
Gemini官方AGY low，K3既有LiteLLM low，各2并发12RPM；首个实际任务校验返回身份/图片读取后继续。
帽：196次AGY CLI+84次客户端HTTP，无有效错答、格式、技术重刷，墙钟4小时，最多8crop/9步。
AGY内部服务端请求不可观测。任一路首题传输失败则停止该路；PAUSE/冻结源变化/额度认证429/
身份异常/连续3技术失败停止派发，保存在途回执，不自动重启。另一正常分支可继续。
生成后逐图检查事实/对象绑定/失败说明/自主恢复/裁图增益；允许原证据充分后直接commit，
但不能计作重新定位成功。可靠搜索miss可保留，未修复关键错误不放行。四档独立保存。
完成CPU预处理和真实token mask验证才交付Database I候选；未收录题仍用原完整路径，
两库任务/组完全相同，不把变体当新独立任务。模型审查不是人工验收，也不证明学生能力。
停止点：本批可执行部分的生成、核验与文档；不扩200题、不训练、不恢复任何旧任务。
'''
    (out/'WORK_PACKAGE.zh.md').write_text(brief)
    return status(out)


def verify(out):
    cfg = json.loads((out/'config.json').read_text())
    if sha(out/'config.json') != (out/'CONFIG.sha256').read_text().strip():
        raise ValueError('Config changed')
    for p, expected in cfg['frozen_files'].items():
        if sha(p) != expected:
            raise ValueError('Frozen source changed: '+p)
    packets = [json.loads(p.read_text()) for p in sorted((out/'inputs').glob('*.json'))]
    if len(packets) != cfg['tasks']:
        raise ValueError('Task count mismatch')
    return cfg, packets


def execute_crop(full, views, action, n, odir):
    W,H = full.size
    k = resolve_source(action['source'], len(views))
    if k is None:
        raise ValueError('Unavailable source')
    x0,y0,x1,y1 = views[k]['effective_box_px']; b = action['bbox']
    req = (x0+b[0]*(x1-x0),y0+b[1]*(y1-y0),x0+b[2]*(x1-x0),y0+b[3]*(y1-y0))
    box, expanded = expand_min_crop(req,W,H,56)
    box = tuple(float(round(x)) for x in box)
    crop = full.crop(tuple(int(x) for x in box))
    path = save_obs(crop, odir/f'obs{len(views)}.jpg')
    vr = view_record('observation_'+str(len(views)),action['source'],box,crop,path,W,H)
    return vr, dict(step=n,source_index=k,requested_box=list(req),expanded=expanded,
                    observation=vr['view_id'])


def as_messages(ro, api, question):
    """Only question, actual received views, and executed assistant turns."""
    views = ro['views']; h,t = proto.first_user_text(question, (),views[0]['source_px'],views[0]['shown_px'],8,True)
    msgs=[text_msg('system',proto.to_api_tags(proto.SYSTEM)),
          dict(role='user',content=[dict(type='text',text=h),api.image(views[0]['path']),dict(type='text',text=t)])]
    available=['original_image']
    for s in ro['steps']:
        msgs.append(text_msg('assistant',proto.to_api_tags(s['parsed']['text'])))
        if s.get('observation'):
            v=next(v for v in views if v['view_id']==s['observation']);available.append(v['view_id'])
            h,t=proto.receipt_text(s['step'],len(available)-1,v['parent'],v['fov_original'],v['source_px'],v['shown_px'],available,len(available)-1,8)
            msgs.append(dict(role='user',content=[dict(type='text',text=h),api.image(v['path']),dict(type='text',text=t)]))
    return msgs


def student_export(ro, question):
    turns=[]; available=['original_image']
    for s in ro['steps']:
        turns.append(dict(role='gpt',text=s['parsed']['text']))
        if s.get('observation'):
            v=next(v for v in ro['views'] if v['view_id']==s['observation']);available.append(v['view_id'])
            h,t=proto.receipt_text(s['step'],len(available)-1,v['parent'],v['fov_original'],v['source_px'],v['shown_px'],available,len(available)-1,8)
            turns.append(dict(role='human',head=h,tail=t))
    e=proto.to_sharegpt(question,turns,[v['path'] for v in ro['views']],ro['views'][0]['source_px'],ro['views'][0]['shown_px'],8)
    e['assistant_supervision']=[s['origin']!='editor_forced' for s in ro['steps']]
    return e


def generate(out, packet, api):
    odir=out/api.role/'obs'/packet['audit_id'];odir.mkdir(parents=True,exist_ok=False)
    if sha(packet['image'])!=packet['source_image_sha256']:
        write_json(out/'PAUSE',dict(reason='source_image_changed',audit_id=packet['audit_id']))
        raise BudgetStop('Source image changed')
    views=copy.deepcopy(packet['prefix_views'])
    for i,v in enumerate(views):
        if sha(v['path'])!=v['sha256']:
            write_json(out/'PAUSE',dict(reason='prefix_image_changed',audit_id=packet['audit_id']))
            raise BudgetStop('Prefix image changed')
        dest=odir/f'obs{i}.jpg';shutil.copyfile(v['path'],dest);v['path']=str(dest)
    first=proto.parse_step(packet['prefix_assistant'])
    ro=dict(protocol=proto.PROTOCOL_VERSION,budget=8,views=views,steps=[dict(step=1,parsed=first,
            observation='observation_1',crops_before=0,available_before=['original_image'],origin='original_prefix',violations=[],flags=proto.soft_flags(first))])
    action=packet['perturbation']; source=action['source']
    forced='<think>\nStep 2 | re_examine\nI will inspect an alternative framing from ['+source+'].\n</think>\n<grounding>'+json.dumps(dict(bbox_2d=action['bbox'],source=source),separators=(',',':'))+'</grounding>'
    p=proto.parse_step(forced)
    if proto.check_step(p,2,1,['original_image','observation_1'],8):
        raise ValueError('Editor action is illegal')
    with Image.open(packet['image']) as im:
        full=im.convert('RGB')
    vr,meta=execute_crop(full,views,p,2,odir);views.append(vr)
    ro['steps'].append(dict(meta,parsed=p,crops_before=1,available_before=['original_image','observation_1'],
                           origin='editor_forced',violations=[],flags=[],request_sha256=None))
    ro['termination']='not_finished';start=time.time()
    write_json(out/api.role/'checkpoints'/(packet['audit_id']+'.json'),dict(rollout=ro,next_step=3))
    for n in range(3,10):
        available=[v['view_id'] for v in views];crops=len(views)-1
        msgs=as_messages(ro,api,packet['question'])
        r=api.call(packet['audit_id']+f'/s{n}',msgs,replicate_id=VERSION+'#'+packet['audit_id'],temperature=.7,top_p=.9)
        raw=r.get('content') or ''
        s=dict(step=n,request_sha256=r['request_sha256'],finish_reason=r['finish_reason'],raw_text=raw,
               crops_before=crops,available_before=available,origin='model_continuation',usage=r.get('usage'))
        ro['steps'].append(s)
        if r['finish_reason']!='stop':
            ro['termination']='api_error' if r['finish_reason']=='error' else 'provider_finish_error';break
        p=proto.parse_step(raw);viol=proto.check_step(p,n,crops,available,8)
        s.update(parsed=p,violations=viol,flags=proto.soft_flags(p))
        # No free-form cleanup, extra-action truncation, or content repairs.
        viol.extend(check_turn(p['text']))
        if viol:
            ro['termination']='protocol_violation';break
        if p['kind']=='answer':
            ro['termination']='abstain' if p['answer'].upper()=='UNCLEAR' else 'answer';ro['answer']=p['answer'];break
        vr,meta=execute_crop(full,views,p,n,odir);s.update(meta);views.append(vr)
        if any(v['effective_box_px']==vr['effective_box_px'] for v in views[:-1]):s['flags'].append('repeat_crop')
        write_json(out/api.role/'checkpoints'/(packet['audit_id']+'.json'),dict(rollout=ro,next_step=n+1))
    ro.update(crops=len(views)-1,answer=ro.get('answer'),export=None)
    if ro['termination'] in ('answer','abstain'):
        ro['export']=student_export(ro,packet['question'])
    errors=[]
    for s in ro['steps']:
        errors.extend(s.get('violations',[]));errors.extend(f for f in s.get('flags',[]) if f in ('coordinate_talk','coordinate_unresolved'))
    result=dict(audit_id=packet['audit_id'],item_id=packet['item_id'],assigned_role=api.role,
                model=api.model,effort=api.spec['effort'],rollout=ro,intervention=packet['intervention'],
                perturbation=packet['perturbation'],cpu=dict(passed=not errors and ro['termination']=='answer',reasons=sorted(set(errors))),
                elapsed_s=round(time.time()-start,2),semantic_review='NOT_REVIEWED',
                config_sha256=sha(out/'config.json'),provenance='original-prefix + editor-forced crop + same-family fresh continuation')
    write_json(out/api.role/'results'/(packet['audit_id']+'.json'),result)
    return result


def status(out):
    cfg=json.loads((out/'config.json').read_text());by={}
    rows=read_rows(out/'budget/dispatch.jsonl') if (out/'budget/dispatch.jsonl').exists() else []
    for role,spec in cfg['models'].items():
        results=[json.loads(p.read_text()) for p in (out/role/'results').glob('*.json')]
        calls=[json.loads(p.read_text()) for p in (out/role/'calls').glob('*.json')]
        issued=[r for r in rows if r['model']==spec['model']];settled={c['request_sha256'] for c in calls}
        by[role]=dict(planned=spec['tasks'],saved=len(results),cpu_candidates=sum(r['cpu']['passed'] for r in results),
                      terminations=dict(Counter(r['rollout']['termination'] for r in results)),
                      dispatched=len(issued),settled=len(calls),pending=[r['request_sha256'] for r in issued if r['request_sha256'] not in settled],
                      request_errors=sum(c['finish_reason']=='error' for c in calls),
                      returned_models=dict(Counter(str(c.get('returned_model')) for c in calls)),
                      stopped=(out/'budget'/('STOP_'+spec['model'])).exists())
    state=json.loads((out/'RUN_STATE.json').read_text()) if (out/'RUN_STATE.json').exists() else dict(state='prepared')
    data=dict(updated_at=time.time(),run_state=state,generators=by,saved=sum(v['saved'] for v in by.values()),
              semantic_KEEP=0,accepted_variants=0,baseline_unchanged_T_Q_G=[4040,4040,3690])
    write_json(out/'STATUS.json',data);return data


def run(out):
    cfg,packets=verify(out)
    with (out/'RUN.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/'RUN_STATE.json').exists():raise ValueError('No automatic restart; reconcile attempts before authorized resume')
        ledger=Ledger(out/'budget',{s['model']:s['call_cap'] for s in cfg['models'].values()})
        apis={r:RecoveryTransport(out,r,cfg,ledger) for r in cfg['models']}
        queues={r:deque(p for p in packets if p['assigned_role']==r) for r in apis}
        write_json(out/'RUN_STATE.json',dict(state='running',pid=os.getpid(),started_at=time.time()))
        done=threading.Event();qlock=threading.Lock();errors=[]
        def watch():
            while not done.wait(15):status(out)
        watcher=threading.Thread(target=watch,daemon=True);watcher.start()
        def one(role,packet):
            try:
                apis[role].check();r=generate(out,packet,apis[role])
                print(json.dumps(dict(audit_id=r['audit_id'],model=r['model'],termination=r['rollout']['termination'],crops=r['rollout']['crops'],cpu=r['cpu']['passed'])),flush=True)
                return r
            except Exception as exc:
                event=dict(audit_id=packet['audit_id'],role=role,error_type=type(exc).__name__)
                errors.append(event);print(json.dumps(event),flush=True)
                ledger.stop(apis[role].model,'unexpected episode exception; preserve checkpoint and reconcile before resume')
                return None
        def smoke(role):
            if not queues[role]:return
            r=one(role,queues[role].popleft())
            if r is None or r['rollout']['termination'] in ('api_error','provider_finish_error'):
                ledger.stop(apis[role].model,'first actual task transport check failed')
        def worker(role):
            while True:
                try:apis[role].check()
                except BudgetStop:return
                with qlock:
                    if not queues[role]:return
                    p=queues[role].popleft()
                one(role,p)
        try:
            with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(smoke,apis))
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures=[pool.submit(worker,r) for r,s in cfg['models'].items() for _ in range(s['workers'])]
                for f in futures:f.result()
            s=status(out)
            write_json(out/'RUN_STATE.json',dict(state='generation_complete_pending_review' if s['saved']==cfg['tasks'] else 'stopped_incomplete',finished_at=time.time(),errors=errors))
        finally:
            done.set();watcher.join(timeout=2);status(out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','verify','run','status']);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();out=a.out.resolve()
    if a.action=='prepare':print(json.dumps(prepare(out),ensure_ascii=False))
    elif a.action=='verify':cfg,items=verify(out);print(json.dumps(dict(valid=True,tasks=len(items))))
    elif a.action=='run':run(out)
    else:print(json.dumps(status(out),ensure_ascii=False))
