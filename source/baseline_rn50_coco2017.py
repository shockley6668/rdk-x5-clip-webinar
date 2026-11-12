"""Paired OpenAI RN50 FP32 retrieval baseline on COCO2017 split."""
import json
import sys
import time
from pathlib import Path
import numpy as np
import onnxruntime as ort
from PIL import Image

home=Path(__file__).resolve().parent
project=home/'clip-photo-search'
dataset=home/'datasets/coco2017_subset'
sys.path.insert(0,str(project/'upstream-openai-clip'))
import clip

entries=json.loads((dataset/'split.json').read_text(encoding='utf8'))['evaluation']
_,preprocess=clip.load(str(project/'RN50.pt'),device='cpu',jit=False)
options=ort.SessionOptions(); options.intra_op_num_threads=8
image_session=ort.InferenceSession(str(project/'float_models/clip_rn50_image.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
text_session=ort.InferenceSession(str(project/'float_models/clip_rn50_text.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
out=home/'results/coco2017'; out.mkdir(parents=True,exist_ok=True)
vectors=[]; start=time.perf_counter()
for i,entry in enumerate(entries):
    image=Image.open(dataset/'evaluation'/entry['file_name']).convert('RGB')
    tensor=preprocess(image).unsqueeze(0).numpy().astype(np.float32)
    v=image_session.run(None,{'image':tensor})[0].reshape(-1)
    vectors.append(v/np.linalg.norm(v))
    if (i+1)%50==0: print('RN50 images',i+1,'/',len(entries),flush=True)
images=np.stack(vectors)
texts=[]
for i,entry in enumerate(entries):
    tokens=clip.tokenize([entry['caption']],truncate=True).numpy()
    texts.append(text_session.run(None,{'texts':tokens})[0])
    if (i+1)%100==0: print('RN50 captions',i+1,'/',len(entries),flush=True)
texts=np.concatenate(texts,axis=0)
texts/=np.linalg.norm(texts,axis=1,keepdims=True)
order=np.argsort(-(texts@images.T),axis=1)
truth=np.arange(len(entries))[:,None]
metrics={f'recall_at_{k}':float(np.mean(np.any(order[:,:k]==truth,axis=1))) for k in (1,5,10)}
np.savez_compressed(out/'rn50_baseline_fp32.npz',image_vectors=images,text_vectors=texts,names=np.array([x['file_name'] for x in entries]))
(out/'rn50_baseline_fp32.json').write_text(json.dumps({'count':len(entries),'seconds':time.perf_counter()-start,'metrics':metrics},indent=2))
print(json.dumps(metrics),flush=True)
