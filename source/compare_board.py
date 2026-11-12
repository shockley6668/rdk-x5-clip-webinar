"""Compare board outputs with matched FP32 image embeddings and text retrieval."""
import argparse
import json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser(); p.add_argument('--baseline',required=True); p.add_argument('--board',required=True); p.add_argument('--out',required=True); a=p.parse_args()
f=np.load(a.baseline,allow_pickle=False); q=np.load(a.board,allow_pickle=False)
names=f['names'].tolist(); order={name:i for i,name in enumerate(q['names'].tolist())}
missing=[n for n in names if n not in order]
if missing: raise ValueError(f'{len(missing)} missing board results')
image=q['vectors'][[order[n] for n in names]].astype(np.float32)
image/=np.linalg.norm(image,axis=1,keepdims=True)
reference=f['image_vectors'] if 'image_vectors' in f else f['vectors']
reference=reference/np.linalg.norm(reference,axis=1,keepdims=True)
cos=np.sum(image*reference,axis=1)
texts=f['text_vectors']; texts=texts/np.linalg.norm(texts,axis=1,keepdims=True)
scores=texts@image.T
baseline_scores=texts@reference.T
results={'images':len(names),'feature_cosine':{'mean':float(np.mean(cos)),'min':float(np.min(cos)),'p05':float(np.percentile(cos,5))},'board_forward_ms':{'median':float(np.median(q['seconds'])*1000),'p95':float(np.percentile(q['seconds'],95)*1000)}}
if len(texts)==len(names):
    truth=np.arange(len(names))[:,None]
    for label,score in [('board',scores),('float',baseline_scores)]:
        ranking=np.argsort(-score,axis=1)
        results[label]={f'recall_at_{k}':float(np.mean(np.any(ranking[:,:k]==truth,axis=1))) for k in (1,5,10)}
    results['top1_same']=float(np.mean(np.argmax(scores,axis=1)==np.argmax(baseline_scores,axis=1)))
Path(a.out).write_text(json.dumps(results,indent=2),encoding='utf8')
print(json.dumps(results,indent=2))
