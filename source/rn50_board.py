"""Execute paired OpenAI RN50 CLIP on RDK X5 and expose local photo search."""
import argparse
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

import cv2
import hbm_runtime
import numpy as np
import onnxruntime as ort

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'sample/runtime/python'))
from simple_tokenizer import SimpleTokenizer
from app import PAGE

MEAN=np.array([0.48145466,0.4578275,0.40821073],np.float32).reshape(3,1,1)
STD=np.array([0.26862954,0.26130258,0.27577711],np.float32).reshape(3,1,1)

class Backend:
    def __init__(self,image_model,text_model):
        self.images=hbm_runtime.HB_HBMRuntime(str(image_model))
        self.name=self.images.model_names[0]
        self.input_name=self.images.input_names[self.name][0]
        self.output_name=self.images.output_names[self.name][0]
        options=ort.SessionOptions(); options.intra_op_num_threads=4
        self.text=ort.InferenceSession(str(text_model),sess_options=options,providers=['CPUExecutionProvider'])
        self.tokenizer=SimpleTokenizer()
    def preprocess(self,path):
        image=cv2.imread(str(path))
        if image is None: raise ValueError('Unreadable image: '+path.name)
        rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        h,w=rgb.shape[:2]
        if h<w: new_h,new_w=224,round(w*224/h)
        else: new_w,new_h=224,round(h*224/w)
        rgb=cv2.resize(rgb,(new_w,new_h),interpolation=cv2.INTER_CUBIC)
        y=(new_h-224)//2; x=(new_w-224)//2; rgb=rgb[y:y+224,x:x+224]
        tensor=rgb.astype(np.float32).transpose(2,0,1)/255.0
        tensor=(tensor-MEAN)/STD
        return tensor[None].astype(np.float32)
    def encode_image(self,path):
        result=self.images.run({self.name:{self.input_name:self.preprocess(path)}})[self.name][self.output_name]
        vector=np.asarray(result,np.float32).reshape(-1); return vector/np.linalg.norm(vector)
    def encode_text(self,query):
        text=self.tokenizer.encoder
        ids=[text['<|startoftext|>']]+self.tokenizer.encode(query)+[text['<|endoftext|>']]
        if len(ids)>77: ids=ids[:76]+[text['<|endoftext|>']]
        tokens=np.zeros((1,77),np.int32);tokens[0,:len(ids)]=ids
        vector=self.text.run(None,{'texts':tokens})[0].reshape(-1)
        return vector/np.linalg.norm(vector)

def index(args):
    model=Backend(args.image_model,args.text_model); root=Path(args.photos).resolve()
    paths=sorted(p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in {'.jpg','.jpeg','.png'})
    vectors=[];names=[];times=[];begin=time.perf_counter()
    for i,path in enumerate(paths,1):
        started=time.perf_counter(); vectors.append(model.encode_image(path)); times.append(time.perf_counter()-started); names.append(path.relative_to(root).as_posix())
        if i%25==0: print(f'Indexed {i}/{len(paths)}',flush=True)
    if not vectors: raise RuntimeError('No images found')
    dest=Path(args.index); dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open('wb') as f: np.savez_compressed(f,vectors=np.stack(vectors),paths=np.array(names),seconds=np.array(times),root=str(root))
    print(f'Indexed {len(names)} in {time.perf_counter()-begin:.1f}s; median {np.median(times)*1000:.1f} ms',flush=True)

def serve(args):
    model=Backend(args.image_model,args.text_model)
    with np.load(args.index,allow_pickle=False) as data:
        vectors=data['vectors']; names=data['paths'].tolist(); root=Path(str(data['root'])).resolve()
    class Handler(BaseHTTPRequestHandler):
        def respond(self,status,kind,data):
            self.send_response(status);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
        def json(self,status,payload): self.respond(status,'application/json',json.dumps(payload).encode())
        def do_GET(self):
            parsed=urlparse(self.path)
            if parsed.path=='/': return self.respond(200,'text/html; charset=utf-8',PAGE.encode())
            if parsed.path.startswith('/photo/'):
                candidate=(root/unquote(parsed.path[len('/photo/'):])).resolve()
                if not candidate.is_relative_to(root) or not candidate.is_file() or candidate.suffix.lower() not in {'.jpg','.jpeg','.png'}: return self.respond(404,'text/plain',b'Not found')
                return self.respond(200,'image/png' if candidate.suffix.lower()=='.png' else 'image/jpeg',candidate.read_bytes())
            if parsed.path!='/api/search': return self.respond(404,'text/plain',b'Not found')
            try:
                query=parse_qs(parsed.query).get('q',[''])[0].strip()
                if not query or len(query)>300: raise ValueError('Enter a short search description')
                started=time.perf_counter(); text=model.encode_text(query); scores=vectors@text; ids=np.argsort(-scores)[:12]
                payload={'results':[{'path':names[i],'score':float(scores[i]),'url':'/photo/'+quote(names[i])} for i in ids],'seconds':time.perf_counter()-started}
                self.json(200,payload)
            except Exception as exc: self.json(400,{'error':str(exc)})
    server=ThreadingHTTPServer((args.host,args.port),Handler)
    print(f'RN50 search: http://{args.host}:{args.port} ({len(names)} images)',flush=True)
    server.serve_forever()

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('action',choices=['index','serve']);p.add_argument('--image-model',required=True);p.add_argument('--text-model',required=True);p.add_argument('--photos',default='coco128/images/train2017');p.add_argument('--index',default='rn50-index.npz');p.add_argument('--host',default='0.0.0.0');p.add_argument('--port',type=int,default=8081)
    args=p.parse_args(); index(args) if args.action=='index' else serve(args)
