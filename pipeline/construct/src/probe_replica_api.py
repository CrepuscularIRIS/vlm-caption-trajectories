"""Adapt probe_api transport for independent replicates and exact PNG delivery."""
import base64
import fcntl
import hashlib
import json
import time
from functools import lru_cache
from pathlib import Path

import requests
from PIL import Image
from probe_api import API,digest


class ReplicaAPI(API):
    @lru_cache(maxsize=16)
    def image(self,path):
        data=Path(path).read_bytes()
        with Image.open(path) as im:mime=Image.MIME[im.format]
        sha=hashlib.sha256(data).hexdigest()
        # Do not silently compress high-resolution arms above the old 8MB limit.
        dest=self.out/'media'/f'{sha}.png'
        with self.lock:
            if not dest.exists():dest.write_bytes(data)
        return {'type':'image_url','image_url':{'url':f'data:{mime};base64,'+base64.b64encode(data).decode()}}

    def call(self,label,msgs,max_tokens=None,*,replicate_id,temperature=0.,top_p=None):
        safe,images=self.safe_messages(msgs)
        payload={'model':self.model,'messages':msgs,'temperature':temperature,
                 'max_tokens':max_tokens or self.max_tokens}
        if top_p is not None:payload['top_p']=top_p
        signature={**payload,'messages':safe,'replicate_id':replicate_id}
        key=digest(signature)
        with (self.out/'locks'/(key+'.lock')).open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            dest=self.out/'calls'/(key+'.json')
            if dest.exists():
                old=json.loads(dest.read_text())
                if old.get('finish_reason') not in ('error',None):return old
            if self.blocked.is_set():raise RuntimeError('Authentication/quota circuit open')
            errors=[];result=None;start=time.monotonic()
            for attempt in range(4):
                self.limiter.wait()
                try:
                    response=requests.post(self.base+'/v1/chat/completions',json=payload,
                        headers={'Authorization':'Bearer '+self.key},timeout=(20,180))
                    if response.status_code!=200:
                        errors.append({'status':response.status_code,'attempt':attempt+1})
                        if response.status_code in (401,402,403):self.blocked.set()
                        if response.status_code in (400,401,402,403,404,413):break
                        time.sleep(min(30,2**(attempt+1)));continue
                    data=response.json();choice=data['choices'][0];message=choice['message']
                    result={'content':message.get('content') or '',
                        'reasoning_content':message.get('reasoning_content') or '',
                        'finish_reason':choice.get('finish_reason'),'returned_model':data.get('model'),
                        'usage':data.get('usage',{}),'response_id':data.get('id')}
                    break
                except (requests.RequestException,ValueError,KeyError,IndexError) as exc:
                    errors.append({'error_type':type(exc).__name__,'attempt':attempt+1})
                    time.sleep(min(20,2**attempt))
            if result is None:result={'content':'','reasoning_content':'','finish_reason':'error',
                                     'returned_model':None,'usage':{}}
            result.update(label=label,request=signature,request_sha256=key,replicate_id=replicate_id,
                images=images,errors=errors,elapsed_s=round(time.monotonic()-start,3))
            if dest.exists():dest.with_name(dest.stem+'.attempt-'+str(time.time_ns())+'.json').write_bytes(dest.read_bytes())
            tmp=dest.with_suffix('.tmp');tmp.write_text(json.dumps(result,ensure_ascii=False));tmp.replace(dest)
            return result
