#!/usr/bin/env python3
"""Loopback-only viewer for legitimately held private questions and actual views.
Never uploads or copies image assets. Only manifest-listed views are served.
"""
import argparse,http.server,json,mimetypes,re,socketserver
from pathlib import Path
from urllib.parse import unquote,urlparse
PUB=Path(__file__).resolve().parents[1]
def main():
 a=argparse.ArgumentParser();a.add_argument('--workspace-root',type=Path,required=True);a.add_argument('--dataset-dir',type=Path,required=True);a.add_argument('--port',type=int,default=0);a.add_argument('--receipt',type=Path);args=a.parse_args()
 root=args.workspace_root.resolve();dd=args.dataset_dir.resolve();manifest=[json.loads(l) for l in (dd/'MANIFEST.jsonl').read_text().splitlines()]
 data=json.loads((dd/'caption_sft_v2_2_keep.json').read_text());assert len(manifest)==len(data)==4040
 records={m['audit_id']:(m,d) for m,d in zip(manifest,data)};assert len(records)==4040
 assert all(len(d['images'])==m['steps']==m['crops']+1 for m,d in records.values())
 assert all(Path(im).is_file() for d in data for im in d['images'])
 class Handler(http.server.BaseHTTPRequestHandler):
  def log_message(self,*args):pass
  def send_data(self,body,kind='application/json; charset=utf-8',status=200):
   self.send_response(status);self.send_header('Content-Type',kind);self.send_header('X-Content-Type-Options','nosniff');self.send_header('Cache-Control','no-store' if self.path.startswith('/private/') else 'max-age=60');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
  def do_GET(self):
   path=unquote(urlparse(self.path).path)
   if path=='/private/status.json':return self.send_data(b'{"local":true,"paths":4040}')
   mt=re.fullmatch(r'/private/([A-Z]\d+)\.json',path)
   if mt:
    if mt[1] not in records:return self.send_error(404)
    m,_=records[mt[1]];p=Path(m['result_path']).resolve()
    if not p.is_relative_to(root):return self.send_error(403)
    row=json.loads(p.read_text());q=row.get('item',{}).get('question')
    if not q:return self.send_error(404,'Question not present in source')
    return self.send_data(json.dumps({'question':q},ensure_ascii=False).encode())
   mt=re.fullmatch(r'/assets/([A-Z]\d+)/(\d+)\.jpg',path)
   if mt:
    if mt[1] not in records:return self.send_error(404)
    _,d=records[mt[1]];i=int(mt[2])
    if i>=len(d['images']):return self.send_error(404)
    p=Path(d['images'][i]);return self.send_data(p.read_bytes(),mimetypes.guess_type(p.name)[0] or 'application/octet-stream')
   target=(PUB/'web'/('index.html' if path=='/' else path.lstrip('/'))).resolve()
   if not target.is_relative_to((PUB/'web').resolve()) or not target.is_file():return self.send_error(404)
   return self.send_data(target.read_bytes(),mimetypes.guess_type(target.name)[0] or 'application/octet-stream')
 class Server(socketserver.ThreadingMixIn,http.server.HTTPServer):daemon_threads=True
 server=Server(('127.0.0.1',args.port),Handler);url=f'http://127.0.0.1:{server.server_port}/?local=1'
 receipt={'url':url,'bound_host':'127.0.0.1','paths':len(records),'actual_views':sum(len(d['images']) for d in data),'no_model_calls':True,'images_uploaded':False}
 if args.receipt:args.receipt.parent.mkdir(parents=True,exist_ok=True);args.receipt.write_text(json.dumps(receipt,indent=2)+'\n')
 print(json.dumps(receipt),flush=True)
 try:server.serve_forever()
 except KeyboardInterrupt:pass
 finally:server.server_close()
if __name__=='__main__':main()
