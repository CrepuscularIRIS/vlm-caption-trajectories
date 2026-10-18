"""Fact-preserving one-intervention pilot; immutable historical calibration."""
import argparse
import copy
from collections import Counter
import json
from pathlib import Path
import time

from PIL import Image
import interp_recovery_calibration as old
import caption_v51 as proto
from export_protocol_check import check_turn
from probe_api import digest
from tiered_transport import ROOT, BudgetStop, write_json

VERSION = 'interp-recovery-facts-v2-20261011'
CURATION = ROOT/'reports/interp_recovery_v2_20261011/CURATION.json'


def forced_turn(packet):
    text = packet['fact_caption']
    if 'alternative framing' in text or '<' in text or '>' in text:
        raise ValueError('Invalid fact caption')
    action = packet['perturbation']
    turn = '<think>\nStep 2 | re_examine\n'+text+'\n</think>\n<grounding>'+json.dumps(
        dict(bbox_2d=action['bbox'],source=action['source']),separators=(',',':'))+'</grounding>'
    p = proto.parse_step(turn)
    errors = proto.check_step(p,2,1,['original_image','observation_1'],8)+check_turn(p['text'])
    if errors: raise ValueError(errors)
    return p


def prefix_packet(result, row, curated, aid):
    # Explicit input allowlist: no old answer/GT, old action2 or views2+.
    ro=result['rollout']; turns=[c['value'] for c in ro['export']['conversations'] if c['from']=='gpt']
    first=proto.parse_step(turns[0])
    if first['kind']!='grounding' or proto.check_step(first,1,0,['original_image'],8):
        raise ValueError('Invalid original prefix')
    packet=dict(audit_id=aid,item_id=row['item_id'],question=result['item']['question'],image=result['item']['image'],
                source_image_sha256=old.sha(result['item']['image']),assigned_role=row['model'].lower(),
                prefix_assistant=turns[0],prefix_views=copy.deepcopy(ro['views'][:2]),
                intervention=curated['kind'],perturbation=copy.deepcopy(curated['action']),
                fact_caption=curated['caption'],budget=8)
    assert [v['view_id'] for v in packet['prefix_views']]==['original_image','observation_1']
    if curated['view_sha256']!=packet['prefix_views'][1]['sha256']:
        raise ValueError('Curated view mismatch')
    forced_turn(packet)
    return packet


def prepare(out):
    if out.exists(): raise ValueError('Fresh run required')
    rows=old.read_rows(old.BASELINE);assert len(rows)==4040
    curated=json.loads(CURATION.read_text());assert len(curated)==8
    by={r['audit_id']:r for r in rows}
    # A new experiment overlay; does not change existing corpus splits.
    groups={g for r in rows for g in r['canonical_groups']}
    # Match the publicly stated selection hash, independently of outcome.
    import hashlib
    split={g:('train' if int(hashlib.sha256(('paired_sft_20261011|'+g).encode()).hexdigest()[:12],16)%10<8
              else 'dev' if int(hashlib.sha256(('paired_sft_20261011|'+g).encode()).hexdigest()[:12],16)%10==8 else 'test') for g in groups}
    if any(len(r['canonical_groups'])!=1 for r in rows):
        raise ValueError('Multi-group task needs linked-component splitting before construction')
    out.mkdir(parents=True)
    write_json(out/'GROUP_SPLIT_OVERLAY.json',dict(seed='paired_sft_20261011',policy='hash buckets 0..7 train, 8 dev, 9 test',
               scope='new paired experiment overlay; does not change sealed corpus partitions or authorize training',groups=split))
    frozen={str(old.BASELINE):old.sha(old.BASELINE),str(CURATION):old.sha(CURATION),
            str(out/'GROUP_SPLIT_OVERLAY.json'):old.sha(out/'GROUP_SPLIT_OVERLAY.json')}
    packets=[];private=[];seen=set()
    for i,c in enumerate(curated,1):
        row=by[c['baseline_id']]
        assert row['model'] in ('Gemini','K3') and len(row['canonical_groups'])==1
        g=row['canonical_groups'][0]
        assert split[g]=='train' and g not in seen;seen.add(g)
        src=Path(row['result_path']);assert old.sha(src)==row['result_sha256']
        result=json.loads(src.read_text());aid=f'P{i:03d}'
        packet=prefix_packet(result,row,c,aid)
        for v in packet['prefix_views']:
            assert old.sha(v['path'])==v['sha256'];frozen[v['path']]=v['sha256']
        frozen[packet['image']]=packet['source_image_sha256']
        frozen[str(src)]=row['result_sha256']
        p=out/'inputs'/(aid+'.json');write_json(p,packet);frozen[str(p)]=old.sha(p)
        packets.append(packet)
        private.append(dict(audit_id=aid,manifest=row,answer_spec=result['item']['answer_spec'],
                            source_effort=result.get('effort'),fact_curation=c))
    write_json(out/'SELECTION_PRIVATE.json',private)
    frozen[str(out/'SELECTION_PRIVATE.json')]=old.sha(out/'SELECTION_PRIVATE.json')
    files=set(old.transport.base.CODE_FILES)|{'interp_recovery_v2.py','interp_recovery_calibration.py','sft_v2_topup.py','agy_reader.py','export_protocol_check.py','caption_v3.py'}
    frozen.update({str(ROOT/'obs/src'/n):old.sha(ROOT/'obs/src'/n) for n in files})
    counts=Counter(p['assigned_role'] for p in packets);assert counts=={'gemini':4,'k3':4}
    cfg=dict(version=VERSION,created_at=time.time(),deadline=time.time()+3600,authorization='2026-10-11 user: 开始构造一下; bounded first eight-task construction pilot under revised design',
             tasks=8,workers=2,budget_crops=8,examples=[],frozen_files=frozen,baseline_manifest=str(old.BASELINE),
             baseline_T_Q_G=[4040,4040,3690],valid_answer_retries=0,technical_retries=0,format_retries=0,
             fact_preservation='root visual curation of supported prefix facts, binding and uncertainty; no old next-action plan',
             continuation='same named generator family; original effort may differ; no GT/old suffix/future views',
             perturbations='one prefix-visible local or tight crop per task; curation uses no future success box',
             masks='editor turn including reliable facts body+EOS=0; accepted reliable prefix/continuation=1',
             no_gpu=True,no_training=True,no_old_run_resume=True,
             models={r:dict(model='gemini-3.8-flash-low' if r=='gemini' else 'k3-256k',effort='low',route='agy' if r=='gemini' else 'api',
                    endpoint=old.transport.AGY if r=='gemini' else 'http://127.0.0.1:4001',tasks=4,workers=1,rpm=6,call_cap=28,max_tokens=8192,
                    allowed_returned_models=['gemini-3.8-flash-low' if r=='gemini' else 'k3-256k']) for r in ('gemini','k3')},
             budget_units='28 AGY CLI plus 28 K3 HTTP; AGY internal calls unobserved')
    write_json(out/'config.json',cfg);(out/'CONFIG.sha256').write_text(old.sha(out/'config.json')+'\n')
    write_json(out/'SELECTION_STATS.json',dict(tasks=8,unique_groups=len(seen),models=dict(counts),
               materials=dict(Counter(p['manifest']['material'] for p in private)),
               kinds=dict(Counter(p['intervention'] for p in packets)),old_crops=dict(Counter(p['manifest']['crops'] for p in private))))
    (out/'WORK_PACKAGE.zh.md').write_text('''# 事实保留插值 v2：首批8题

用户2026-10-11授权“开始构造一下”。Gemini AGY low 4题、K3 LiteLLM low 4题；
只生成8个冻结任务，每原图组一题，选材仅依据旧基线模型/材料/哈希与可见前缀，非旧校准成败。
从已见obs1及原下一Caption中人工核对并保留当前事实/绑定/不确定性，不保留旧具体下一计划、框或后缀。
手工局部框仅依据已见图像提出，不承诺必然失败。保持原Step1，真实执行一次干预，之后同族模型自主续写。
不提供GT、未来图或旧后缀；mask编辑轮正文与EOS，可靠原前缀/新续写待审核。
各1并发6RPM、28客户端调用帽，共56；一小时、max8crop/9助手步；无有效错答/格式/技术重刷。
沿用传输持久回执与RUN.lock；首题传输失败停该路，额度认证429/身份异常/3连续技术失败/PAUSE/源变停派。
不自动重启未决请求，不换账号模型、不扩题、不训练/GPU/下载或恢复旧任务。
group hash80/10/10是新配对实验overlay，仅本批train桶入选，不改已封存corpus split。
生成后Codex root实际逐图核对原图和全部crop，GT与历史标签分离；四档独立保存，不冒称人工验收。
合格示例原样导出并CPU/mask检查，4B实际loader尚未验收则如实只交付校准候选。
停止点：本批8题的生成、复核、网页和文档；原4040及旧四替换不改，不自动扩200题。
''')
    return old.status(out)


def generate(out,packet,api):
    odir=out/api.role/'obs'/packet['audit_id'];odir.mkdir(parents=True,exist_ok=False)
    if old.sha(packet['image'])!=packet['source_image_sha256']:raise BudgetStop('Source image changed')
    views=copy.deepcopy(packet['prefix_views'])
    for i,v in enumerate(views):
        if old.sha(v['path'])!=v['sha256']:raise BudgetStop('Prefix image changed')
        dest=odir/f'obs{i}.jpg';old.shutil.copyfile(v['path'],dest);v['path']=str(dest)
    first=proto.parse_step(packet['prefix_assistant'])
    ro=dict(protocol=proto.PROTOCOL_VERSION,budget=8,views=views,steps=[dict(step=1,parsed=first,observation='observation_1',
            crops_before=0,available_before=['original_image'],origin='original_prefix',violations=[],flags=proto.soft_flags(first))])
    p=forced_turn(packet)
    with Image.open(packet['image']) as im:full=im.convert('RGB')
    vr,meta=old.execute_crop(full,views,p,2,odir);views.append(vr)
    ro['steps'].append(dict(meta,parsed=p,crops_before=1,available_before=['original_image','observation_1'],
                           origin='editor_forced',violations=[],flags=[],request_sha256=None))
    ro['termination']='not_finished';start=time.time()
    write_json(out/api.role/'checkpoints'/(packet['audit_id']+'.json'),dict(rollout=ro,next_step=3))
    for n in range(3,10):
        available=[v['view_id'] for v in views];crops=len(views)-1
        r=api.call(packet['audit_id']+f'/s{n}',old.as_messages(ro,api,packet['question']),replicate_id=VERSION+'#'+packet['audit_id'],temperature=.7,top_p=.9)
        raw=r.get('content') or ''
        s=dict(step=n,request_sha256=r['request_sha256'],finish_reason=r['finish_reason'],raw_text=raw,
               crops_before=crops,available_before=available,origin='model_continuation',usage=r.get('usage'));ro['steps'].append(s)
        if r['finish_reason']!='stop':ro['termination']='api_error' if r['finish_reason']=='error' else 'provider_finish_error';break
        p=proto.parse_step(raw);viol=proto.check_step(p,n,crops,available,8)+check_turn(p['text'])
        s.update(parsed=p,violations=viol,flags=proto.soft_flags(p))
        if viol:ro['termination']='protocol_violation';break
        if p['kind']=='answer':
            ro['termination']='abstain' if p['answer'].upper()=='UNCLEAR' else 'answer';ro['answer']=p['answer'];break
        vr,meta=old.execute_crop(full,views,p,n,odir);s.update(meta);views.append(vr)
        if any(v['effective_box_px']==vr['effective_box_px'] for v in views[:-1]):s['flags'].append('repeat_crop')
        write_json(out/api.role/'checkpoints'/(packet['audit_id']+'.json'),dict(rollout=ro,next_step=n+1))
    ro.update(crops=len(views)-1,answer=ro.get('answer'),export=None)
    if ro['termination'] in ('answer','abstain'):ro['export']=old.student_export(ro,packet['question'])
    errors=[v for s in ro['steps'] for v in s.get('violations',[])]
    errors += [f for s in ro['steps'] for f in s.get('flags',[]) if f in ('coordinate_talk','coordinate_unresolved')]
    result=dict(audit_id=packet['audit_id'],item_id=packet['item_id'],assigned_role=api.role,model=api.model,effort=api.spec['effort'],
                rollout=ro,intervention=packet['intervention'],perturbation=packet['perturbation'],
                cpu=dict(passed=not errors and ro['termination']=='answer',reasons=sorted(set(errors))),elapsed_s=round(time.time()-start,2),
                semantic_review='NOT_REVIEWED',config_sha256=old.sha(out/'config.json'),
                provenance='unchanged original first step + visually curated prefix facts in masked intervention + autonomous same-family continuation')
    write_json(out/api.role/'results'/(packet['audit_id']+'.json'),result);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','verify','run','status']);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();out=a.out.resolve()
    if a.action=='prepare':print(json.dumps(prepare(out),ensure_ascii=False))
    else:
        cfg,items=old.verify(out)
        if cfg['version']!=VERSION:raise ValueError('Only this fresh version may run')
        if a.action=='verify':print(json.dumps(dict(valid=True,tasks=len(items))))
        elif a.action=='status':print(json.dumps(old.status(out)))
        else:
            old.generate=generate  # Runtime reuse of lock/queue/receipt code; old file stays immutable.
            old.run(out)


if __name__=='__main__':main()
