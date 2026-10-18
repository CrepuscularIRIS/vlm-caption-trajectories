"""Remaining untouched full-batch tasks with prefix-local critical-patch omission.

The first construction's receipts remain immutable. A visible-patch annotation
is a hypothesis, never a guarantee or a future success box. No old attempt retries.
"""
import argparse
import copy
import json
import math
from collections import Counter
from pathlib import Path
import re
import time

import interp_recovery_full as full
from monitor_step import expand_min_crop
from tiered_transport import ROOT, write_json

VERSION='interp-recovery-full-critical-visible-patch-r2-20261011'
PARENT=ROOT/'obs/runs/interp_recovery_full_20261011'
EDITOR_SYSTEM='''You annotate ONE question-critical detail ALREADY VISIBLE in observation_1.
Inputs are only the question, executed first crop, original shown image and observation_1.
No GT, future evidence, old next action or final answer is available. Do NOT solve from memory.
Return current visually supported facts and instance bindings in 1-3 English sentences.
Preserve already readable task-relevant facts; do not erase them or invent uncertainty
just to make the intervention harder. Cite [original_image] or [observation_1].
The caption must contain ONLY current facts and genuine uncertainty. NO editor, crop-plan,
plausibility, hypothetical intervention, or prediction of what a future crop might lose.
Separately annotate target_bbox normalized to observation_1: the SMALL concrete visual
feature essential for answering this question (e.g. requested word, birth year, distinguishing
attribute, row label). It must be visible in observation_1. It is NOT a suggested action.
Do NOT choose decorative context irrelevant to the question, an entire large page, or a future
successful crop. The environment will deterministically crop a NEARBY region avoiding this
currently observed feature. The continuation receives no target_bbox or return-to-target instruction.
If no localized critical feature is visible/definable, set applicable=false; do not fabricate it.
Return exactly JSON keys:
{"applicable":true,"caption":"[observation_1] ...",
 "target_bbox":[0.2,0.3,0.6,0.4],
 "expected_missing_evidence":"name the question-critical visible feature",
 "reason":"explain its relevance to this question, not decorations"}.
For applicable=false, target_bbox may be null. No answer letter, GT or future-view fields.'''


def adjacent_omission(packet,target):
    """Choose once from neighboring rectangles, testing actual expanded geometry.

    Avoids only an annotated patch in the CURRENT received view. Does not inspect
    any original future box, Caption, answer or review to select the intervention.
    """
    x0,y0,x1,y1=target;tw,th=x1-x0,y1-y0;cx,cy=(x0+x1)/2,(y0+y1)/2
    parent=packet['prefix_views'][1]['effective_box_px'];pw,ph=parent[2]-parent[0],parent[3]-parent[1]
    native=packet['prefix_views'][0]['effective_box_px'];W,H=native[2],native[3]
    w=max(.15,min(.5,tw*1.4),56/pw);h=max(.15,min(.5,th*1.4),56/ph)
    gap=max(.01,2/min(pw,ph))
    boxes=[(cx-w/2,y0-gap-h,cx+w/2,y0-gap),
           (cx-w/2,y1+gap,cx+w/2,y1+gap+h),
           (x0-gap-w,cy-h/2,x0-gap,cy+h/2),
           (x1+gap,cy-h/2,x1+gap+w,cy+h/2)]
    candidates=[]
    critical=[parent[0]+x0*pw,parent[1]+y0*ph,parent[0]+x1*pw,parent[1]+y1*ph]
    critical_area=(critical[2]-critical[0])*(critical[3]-critical[1])
    for box in boxes:
        if not(0<=box[0]<box[2]<=1 and 0<=box[1]<box[3]<=1):continue
        req=[parent[0]+box[0]*pw,parent[1]+box[1]*ph,parent[0]+box[2]*pw,parent[1]+box[3]*ph]
        effective,expanded=expand_min_crop(req,W,H,56)
        intersection=max(0,min(effective[2],critical[2])-max(effective[0],critical[0]))*max(0,min(effective[3],critical[3])-max(effective[1],critical[1]))
        coverage=intersection/critical_area
        if coverage>.02:continue
        candidates.append(dict(source='observation_1',bbox=[round(v,6) for v in box],
                               annotated_patch_coverage=coverage,native_min_expansion=expanded))
    if not candidates:return None
    return min(candidates,key=lambda b:full.digest([VERSION,packet['audit_id'],b['bbox']]))


def validate_proposal(obj,packet):
    keys={'applicable','caption','target_bbox','expected_missing_evidence','reason'}
    if not isinstance(obj,dict) or set(obj)!=keys or type(obj['applicable']) is not bool:
        raise ValueError('Invalid current-critical-patch JSON')
    if not isinstance(obj['reason'],str) or not obj['reason'].strip():raise ValueError('Missing task relevance')
    if not obj['applicable']:return None
    target=obj['target_bbox']
    if not isinstance(target,list) or len(target)!=4 or any(type(v) not in (int,float) or not math.isfinite(v) for v in target):
        raise ValueError('Invalid visible patch')
    if not(0<=target[0]<target[2]<=1 and 0<=target[1]<target[3]<=1):raise ValueError('Invalid visible patch range')
    if (target[2]-target[0])*(target[3]-target[1])>.6:raise ValueError('Patch too broad to be a local discriminating feature')
    cap=obj['caption']
    if not isinstance(cap,str) or not 15<=len(cap)<=700 or not re.search(r'\[(original_image|observation_1)\]',cap):
        raise ValueError('Missing short anchored current facts')
    if re.search(r'\[(?!original_image\]|observation_1\])|<|>|\bStep\s*\d|\b(?:editor|intervention|perturbation|plausibl(?:e|y)|ground\s*truth|correct\s*answer)\b',cap,re.I):
        raise ValueError('Hypothetical editor/answer/future text in factual caption')
    if re.search(r'\b(?:will|would|could|might)\b.{0,50}\b(?:crop|zoom|frame|framing|inspect)\b',cap,re.I):
        raise ValueError('New action plan in fact-only caption')
    if not isinstance(obj['expected_missing_evidence'],str) or not obj['expected_missing_evidence'].strip():
        raise ValueError('Missing question-critical feature')
    action=adjacent_omission(packet,target)
    if action is None:return None
    p=copy.deepcopy(packet)
    p.update(fact_caption=cap,intervention='adjacent_current_critical_patch_omission',perturbation=action,
             critical_patch_bbox=target,proposal_status='NOT_VISUALLY_VERIFIED',proposal_reason=obj['reason'],
             expected_missing_evidence=obj['expected_missing_evidence'],
             geometry_scope='current received obs1 annotated patch only; not semantic proof of failure')
    parsed=full.pilot.forced_turn(p)
    if 'coordinate_talk' in full.proto.soft_flags(parsed):raise ValueError('Coordinates in factual caption')
    return p


def prepare(out):
    if out.exists():raise ValueError('Fresh remaining-only run required')
    cfg,packets=full.verify(PARENT)
    state=json.loads((PARENT/'RUN_STATE.json').read_text())
    if state['state']=='running':raise ValueError('Drain old in-flight requests first')
    for role in cfg['models']:
        for path in (PARENT/role/'attempts').glob('*.json'):
            a=json.loads(path.read_text())
            if a['state']!='settled' or full.old.sha(a['receipt'])!=a['receipt_sha256']:
                raise ValueError('Unsettled/changed old request')
    dispatch=full.old.read_rows(PARENT/'budget/dispatch.jsonl')
    started={d['label'].split('/')[0] for d in dispatch}
    remaining=[p for p in packets if p['audit_id'] not in started]
    private=[r for r in json.loads((PARENT/'SELECTION_PRIVATE.json').read_text()) if r['audit_id'] not in started]
    out.mkdir(parents=True)
    hashes={}
    for p in remaining:
        path=out/'inputs'/(p['audit_id']+'.json');write_json(path,p);hashes[str(path)]=full.old.sha(path)
    write_json(out/'INPUT_HASHES.json',hashes);write_json(out/'SELECTION_PRIVATE.json',private)
    for name in ('GROUP_SPLIT_OVERLAY.json','SCOPE_ALL_4040.json','REUSE_APPROVED.json'):
        (out/name).write_bytes((PARENT/name).read_bytes())
    write_json(out/'PREVIOUS_ATTEMPTS_IMMUTABLE.json',dict(parent=str(PARENT),started_ids=sorted(started),
        dispatched_requests=len(dispatch),saved_results=sum(len(list((PARENT/r/'results').glob('*.json'))) for r in cfg['models']),
        pending_requests=0,remaining=len(remaining),previous_frozen_rejections_not_retried=True))
    counts=Counter(p['assigned_role'] for p in remaining)
    cfg=copy.deepcopy(cfg);cfg.update(version=VERSION,created_at=time.time(),deadline=time.time()+48*3600,
        tasks=len(remaining),parent_run=str(PARENT),total_new_scope_tasks=1509,prior_started_tasks=len(started),
        intervention='one deterministic adjacent omission of a model-annotated CURRENT obs1 critical patch; no future box',
        source_checks='reuse frozen prior full pixel preflight; verify selected prefix/native pixels at each dispatch')
    for role,s in cfg['models'].items():s.update(tasks=counts[role],call_cap=counts[role]*8)
    for path in (Path(__file__).resolve(),out/'INPUT_HASHES.json',out/'SELECTION_PRIVATE.json',out/'GROUP_SPLIT_OVERLAY.json',
                 out/'SCOPE_ALL_4040.json',out/'REUSE_APPROVED.json',out/'PREVIOUS_ATTEMPTS_IMMUTABLE.json',PARENT/'config.json',PARENT/'budget/dispatch.jsonl',PARENT/'RUN_STATE.json',PARENT/'PAUSE'):
        cfg['frozen_files'][str(path)]=full.old.sha(path)
    write_json(out/'config.json',cfg);(out/'CONFIG.sha256').write_text(full.old.sha(out/'config.json')+'\n')
    write_json(out/'SELECTION_STATS.json',dict(new_tasks=len(remaining),prior_started=len(started),total_scope_tasks=1509,
        allocation=dict(counts),old_crops=dict(Counter(r['manifest']['crops'] for r in private)),
        benchmarks=dict(Counter(r['manifest']['bench'] for r in private)),
        no_old_attempt_retries=True,baseline_T_Q_G=[4040,4040,3690]))
    # Parent statistics intentionally describe the earlier method; this run freezes its own rows.
    return full.status(out)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','verify','run','status'])
    parser.add_argument('--out',type=Path,required=True);a=parser.parse_args();out=a.out.resolve()
    if a.action=='prepare':print(json.dumps(prepare(out),ensure_ascii=False))
    else:
        full.VERSION=VERSION;full.EDITOR_SYSTEM=EDITOR_SYSTEM;full.validate_proposal=validate_proposal
        if a.action=='verify':cfg,items=full.verify(out);print(json.dumps(dict(valid=True,tasks=len(items))))
        elif a.action=='run':full.run(out)
        else:print(json.dumps(full.status(out),ensure_ascii=False))
