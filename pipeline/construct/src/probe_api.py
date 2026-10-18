"""API-only, resumable probe transport. No local model or CUDA execution."""
from __future__ import annotations
import base64
import fcntl
import hashlib
import io
import json
import logging
import re
import threading
import time
from pathlib import Path
from functools import lru_cache
import requests
from PIL import Image
from run_api_t0 import RateLimiter, image_b64, load_env

LOG = logging.getLogger('probe')
JSON_INSTR = 'Return only one JSON object: {"answer": "your final answer"}. For multiple choice, answer must be the option letter. Do not put reasoning in the answer field.'

def digest(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False).encode()).hexdigest()

def text_msg(role, text):
    return {'role':role,'content':[{'type':'text','text':text}]}

def structured(raw):
    s=raw.strip()
    if s.startswith('```'): s=re.sub(r'^```(?:json)?\s*|\s*```$','',s,flags=re.I)
    try:
        obj=json.loads(s)
        return obj if isinstance(obj,dict) else None
    except (ValueError,TypeError): return None

def final_answer(reply):
    if reply.get('finish_reason')!='stop': return None
    obj=structured(reply.get('content',''))
    if obj and isinstance(obj.get('answer'),(str,int,float)):
        return str(obj['answer']).strip()
    # Only an entire single-letter answer is an accepted fallback, never intermediate prose.
    m=re.fullmatch(r'\s*\(?([A-Ja-j])\)?[.]?\s*',reply.get('content',''))
    return m.group(1).upper() if m else None

class API:
    def __init__(self,out,model,rpm=42,max_tokens=8192):
        self.base,self.key,_=load_env()
        self.out=Path(out);self.model=model;self.max_tokens=max_tokens
        self.limiter=RateLimiter(rpm);self.lock=threading.Lock();self.blocked=threading.Event()
        for d in ['calls','media','locks']: (self.out/d).mkdir(parents=True,exist_ok=True)

    @lru_cache(maxsize=450)
    def image(self,path):
        b64,_=image_b64(path);data=base64.b64decode(b64)
        with Image.open(io.BytesIO(data)) as im: mime=Image.MIME[im.format]
        sha=hashlib.sha256(data).hexdigest();ext={'image/jpeg':'jpg','image/png':'png','image/webp':'webp'}.get(mime,'bin')
        dest=self.out/'media'/f'{sha}.{ext}'
        with self.lock:
            if not dest.exists(): dest.write_bytes(data)
        return {'type':'image_url','image_url':{'url':f'data:{mime};base64,{b64}'}}

    def safe_messages(self,msgs):
        safe=json.loads(json.dumps(msgs));images=[]
        for msg in safe:
            content=msg.get('content',[])
            if not isinstance(content,list): continue
            for part in content:
                if part.get('type')=='image_url':
                    url=part['image_url']['url'];prefix,enc=url.split(',',1);data=base64.b64decode(enc)
                    sha=hashlib.sha256(data).hexdigest()
                    with Image.open(io.BytesIO(data)) as im: size=list(im.size)
                    images.append({'sha256':sha,'mime':prefix[5:-7],'bytes':len(data),'dimensions':size})
                    part['image_url']['url']='sha256:'+sha
        return safe,images

    def call(self,label,msgs,max_tokens=None):
        safe,_=self.safe_messages(msgs)
        h=digest({'model':self.model,'messages':safe,'temperature':0,'max_tokens':max_tokens or self.max_tokens})
        # flock on separate descriptors serializes threads AND processes per request.
        with (self.out/'locks'/(h+'.lock')).open('a') as guard:
            fcntl.flock(guard,fcntl.LOCK_EX)
            return self._call_locked(label,msgs,max_tokens)

    def _call_locked(self,label,msgs,max_tokens=None):
        safe,images=self.safe_messages(msgs)
        payload={'model':self.model,'messages':msgs,'temperature':0,'max_tokens':max_tokens or self.max_tokens}
        signature={'model':self.model,'messages':safe,'temperature':0,'max_tokens':payload['max_tokens']}
        h=digest(signature);dest=self.out/'calls'/(h+'.json')
        if dest.exists():
            result=json.loads(dest.read_text())
            if result.get('finish_reason') not in ['error',None]:return result
        if self.blocked.is_set():raise RuntimeError('API circuit stopped after authentication/quota error; inspect saved receipts before restart')
        errors=[];start=time.monotonic();result=None
        for attempt in range(4):
            self.limiter.wait()
            try:
                r=requests.post(self.base+'/v1/chat/completions',json=payload,headers={'Authorization':'Bearer '+self.key},timeout=(20,180))
                if r.status_code!=200:
                    errors.append({'status':r.status_code,'attempt':attempt+1})
                    if r.status_code in [401,403]:self.blocked.set()
                    if r.status_code in [400,401,403,404]:break
                    time.sleep(min(30,2**(attempt+1)));continue
                d=r.json();c=d['choices'][0];m=c['message']
                result={'content':m.get('content') or '', 'reasoning_content':m.get('reasoning_content') or '',
                        'finish_reason':c.get('finish_reason'),'returned_model':d.get('model'),
                        'usage':d.get('usage',{}),'response_id':d.get('id')}
                break
            except (requests.RequestException,ValueError,KeyError,IndexError) as e:
                errors.append({'error_type':type(e).__name__,'attempt':attempt+1});time.sleep(min(20,2**attempt))
        if result is None:result={'content':'','finish_reason':'error','returned_model':None,'usage':{}}
        result.update(label=label,request=signature,images=images,request_sha256=h,elapsed_s=round(time.monotonic()-start,3),errors=errors)
        # Preserve every failed attempt receipt when retrying the same request.
        if dest.exists():
            prior=dest.with_name(dest.stem+'.attempt-'+str(time.time_ns())+'.json')
            prior.write_bytes(dest.read_bytes())
        temp=dest.with_suffix('.tmp')
        with self.lock:temp.write_text(json.dumps(result,ensure_ascii=False));temp.replace(dest)
        LOG.info('%s %s %s',self.model,label,result['finish_reason'])
        return result

def response_ref(r):
    return {k:r.get(k) for k in ['request_sha256','finish_reason','returned_model','usage','errors']}|{'answer':final_answer(r)}

def canonical_choice(r,choices):
    a=final_answer(r)
    return a.upper() if a is not None and a.upper() in choices else None

def mc_score(r,gold,choices='ABCD'):
    a=canonical_choice(r,choices)
    return None if a is None else a==gold.strip().upper()
