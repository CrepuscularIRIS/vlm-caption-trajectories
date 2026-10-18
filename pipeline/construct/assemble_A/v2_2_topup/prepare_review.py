"""Prepare blinded, actual-view review and unchanged CPU exports, offline only."""
import collections
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT=Path('${ICML_DIR}');OUT=Path(__file__).resolve().parent
RUN=ROOT/'obs/runs/sft_v2_topup_k3_20261008_r2'
sys.path.insert(0,str(ROOT/'obs/src'))
from export_protocol_check import check_record,HOLD
from corpus_scoring import score

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')

def main():
    if (OUT/'PRIVATE_INDEX.json').exists():raise SystemExit('Prepared already; do not overwrite')
    state=json.loads((RUN/'RUN_STATE.json').read_text())
    assert state['state']=='complete_pending_process_review' and state['saved']==111 and not state['errors']
    obs=json.loads((RUN/'MONITOR_OBSERVATION.json').read_text())
    assert obs['pending']==0 and obs['run_lock']=='free' and not obs['problems'] and not obs['errors']
    for d in ['packets','blind','blind_notes','records','queues','media','cpu_candidate_check']:(OUT/d).mkdir(exist_ok=True)
    rawindex=json.loads((RUN/'SCORING_AND_REVIEW_MAP_PRIVATE.json').read_text())
    eligible=[r for r in rawindex if r['score']['score']==1 and r['cpu']['passed']]
    excluded=[r for r in rawindex if r not in eligible]
    assert len(eligible)==88 and len(excluded)==23
    eligible.sort(key=lambda r:r['review_id'])
    ix=[];checks=[];dataset=[]
    for n,row in enumerate(eligible,1):
        aid=f'R{n:04d}';raw=json.loads(Path(row['result']).read_text());ro=raw['rollout']
        assert sha(row['result'])==row['result_sha256']
        assert score(raw['item']['answer_spec'],ro['answer'])['score']==1
        packets=[]
        for stage,name in [('01_blind_question','blind'),('02_trajectory','packets')]:
            packet=json.loads((RUN/'review_handoff'/row['review_id']/(stage+'.json')).read_text())
            packet.pop('review_id');packet['audit_id']=aid
            for j,v in enumerate(packet['views']):
                dest=OUT/'media'/aid/f'view{j}.jpg';dest.parent.mkdir(exist_ok=True)
                shutil.copyfile(v['image_path'],dest);assert sha(dest)==v['sha256'];v['image_path']=str(dest)
            save(OUT/name/f'{aid}.json',packet);packets.append(packet)
        packet=packets[1]
        row=dict(row,audit_id=aid,item=raw['item'],crops=ro['crops'],steps=len(ro['steps']),packet_sha256=sha(OUT/'packets'/f'{aid}.json'))
        ix.append(row);rec=ro['export'];images=[str((ROOT/p).resolve()) for p in rec['images']]
        errors=[];violations=sorted(set(check_record(rec)))
        if set(violations)&HOLD:errors.append('hard_protocol_failure')
        if [sha(p) for p in images]!=[v['sha256'] for v in packet['views']]:errors.append('image_order_or_hash')
        assistants=[c['value'] for c in rec['conversations'] if c['from']=='gpt']
        expected=[s['parsed']['text'].replace('<caption>','<think>').replace('</caption>','</think>') for s in ro['steps']]
        if [s.strip() for s in assistants]!=[s.strip() for s in expected]:errors.append('visible_target_mutation')
        if len(assistants)!=len(ro['steps']) or len(images)!=ro['crops']+1:errors.append('counts')
        available=['original_image']
        for step in packet['steps']:
            if step['available_before']!=available:errors.append('prefix_order')
            if step['observation']:available.append(step['observation'])
        if available!=[v['view_id'] for v in packet['views']]:errors.append('returned_view_order')
        if raw['item']['split']!='train':errors.append('not_internal_train')
        checks.append(dict(audit_id=aid,idx=n-1,result_sha256=row['result_sha256'],errors=errors,protocol=violations,exact_source_export=not errors))
        dataset.append(dict(conversations=rec['conversations'],images=images))
    save(OUT/'PRIVATE_INDEX.json',ix);save(OUT/'GATE_EXCLUDED_PRIVATE.json',excluded);save(OUT/'EXPORT_CHECKS.json',checks)
    assert not any(x['errors'] for x in checks),checks
    name='caption_final_candidates';p=OUT/'cpu_candidate_check';save(p/f'{name}.json',dataset)
    save(p/'dataset_info.json',{name:dict(file_name=f'{name}.json',formatting='sharegpt',columns={'messages':'conversations','images':'images'},tags={'role_tag':'from','content_tag':'value','user_tag':'human','assistant_tag':'gpt','system_tag':'system'})})
    qs=[[] for _ in range(6)];loads=[0]*6
    for r in sorted(ix,key=lambda x:(-x['crops'],x['audit_id'])):
        w=min(range(6),key=lambda i:(loads[i],len(qs[i]),i));qs[w].append(r['audit_id']);loads[w]+=r['crops']+1
    for w,q in enumerate(qs,1):save(OUT/'queues'/f'Q{w}.json',dict(audit_ids=q,views=loads[w-1]))
    oldbrief=(ROOT/'reports/sft_v2_topup_review_20261008/REVIEW_BRIEF.zh.md').read_text()
    schema=oldbrief[oldbrief.index('结果schema'):oldbrief.index('每完成50条')]
    brief='''# 本批88条新路径逐图审核：唯一范围

你只读本目录blind/Rxxxx.json、packets/Rxxxx.json及其中image_path实际图片。不得读PRIVATE_INDEX、GT、源结果、历史标签、其他worker意见或其他批次。
每题先读blind并通过view_image实际打开原图，写blind_notes/Rxxxx.json短答案或不确定性；完成该盲读记录后才打开同题packet。不得先看到轨迹再声称盲答。
再实际打开全部返回crop，按available_before核查每步证据，不能用未来crop补救先前确定性错误。完整回顾不冒充隔离式前缀验证。
用户明确授权独立Codex native reviewer；这是模型审核，不是人工验收或多族投票。无需外部搜索，专业事实无法可靠核实可UNRESOLVED。

导师和用户规则：关键事实、对象/属性/行列绑定、动作与实际观测一致。原图清晰允许直答；不强制crop或HOLD/REVISE。定位后细节、跨区域比较绑定、诚实未命中后换区有教学价值。每次crop要有用途；几何重叠不自动算无用，但无新增可读性/上下文/排除的重复确认应WEAK。
KEEP：完整路径的关键事实/绑定/答案可靠，所有crop有具体作用。WEAK：核心可靠但冗余或非关键说明瑕疵。BAD：确定的关键错误、错绑、虚构证据或未修复错误前缀。UNRESOLVED：实际图像、题意、指代或关键知识无法确认。不要按选项拼写替代图上的字，不因终答正确放行错误过程。
两阶段查看仍能用原图解决的问题允许直答；静态图合理姿态推断和无法证明身份/未来事件要区分。既有标签不作新判断依据。
你的独占输出为records中自己queue所有Rxxxx.json、blind_notes同ID及queues/Qn_STATUS.json、Qn_DONE.json。你不是唯一工作者，不修改/撤销别人文件。每完成一题立刻落盘，不批量猜标签或声称没看的图已看。
完成自己14–15条即可停止，不扩量、不调外部API、不生成、不训练、不改GT。issue.quote须是对应step原文连续片段；若引用question必须加quote_source:'question'。reviewer填r2_Qn。

'''+schema+'''
完成时检查所有ID、viewed_ids与crop_assessment覆盖及引用逐字一致，写Qn_DONE.json（counts、view_count、完整ID清单），再向root简报。只收KEEP到主集，其他三档独立保留。
'''
    (OUT/'REVIEW_BRIEF.zh.md').write_text(brief)
    for name in ['lf_preprocess_check.py','verify_cached_mask.py']:
        text=(ROOT/'reports/sft_v2_topup_review_20261008'/name).read_text().replace('len(ds)==len(raw)==37','len(ds)==len(raw)==88')
        (OUT/name).write_text(text)
    summary=dict(generated=111,correct=90,CPU_correct=88,requests=292,technical_errors=0,gate_excluded=23,review_views=sum(loads),queues=[dict(tasks=len(q),views=loads[i]) for i,q in enumerate(qs)],no_model_calls=True,no_training=True)
    save(OUT/'PREFLIGHT.json',summary);save(OUT/'GENERATION_VERIFICATION.json',obs);print(json.dumps(summary))

if __name__=='__main__':main()
