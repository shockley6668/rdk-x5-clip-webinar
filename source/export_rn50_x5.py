"""Export paired OpenAI RN50 CLIP image and text encoders at ONNX opset 11."""
import hashlib
import sys
import types
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as F

root=Path(__file__).resolve().parent
sys.path.insert(0,str(root/'upstream-openai-clip'))
import clip

weights=root/'RN50.pt'
expected='afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762'
if hashlib.sha256(weights.read_bytes()).hexdigest()!=expected:
    raise RuntimeError('RN50 checkpoint hash mismatch')
model,preprocess=clip.load(str(weights),device='cpu',jit=False)
model.eval().float()
print('vision',type(model.visual).__name__,'image output',model.visual.output_dim,'embed',model.text_projection.shape[1],flush=True)

def attention(self,x):
    a=self.attn
    length,batch,width=x.shape
    heads=a.num_heads
    dim=width//heads
    q,k,v=F.linear(x,a.in_proj_weight,a.in_proj_bias).chunk(3,dim=-1)
    q=q.reshape(length,batch,heads,dim).permute(1,2,0,3)
    k=k.reshape(length,batch,heads,dim).permute(1,2,0,3)
    v=v.reshape(length,batch,heads,dim).permute(1,2,0,3)
    scores=torch.matmul(q,k.transpose(-1,-2)) * (dim ** -0.5)
    if self.attn_mask is not None:
        scores=scores+self.attn_mask
    weights=torch.softmax(scores,dim=-1)
    result=torch.matmul(weights,v).permute(2,0,1,3).reshape(length,batch,width)
    return F.linear(result,a.out_proj.weight,a.out_proj.bias)

def attn_pool_forward(self,x):
    batch,width,h,w=x.shape
    seq=x.flatten(start_dim=2).permute(2,0,1)
    seq=torch.cat([seq.mean(dim=0,keepdim=True),seq],dim=0)
    seq=seq+self.positional_embedding[:,None,:].to(seq.dtype)
    length,batch,width=seq.shape
    heads=self.num_heads
    dim=width//heads
    q=F.linear(seq[:1],self.q_proj.weight,self.q_proj.bias)
    k=F.linear(seq,self.k_proj.weight,self.k_proj.bias)
    v=F.linear(seq,self.v_proj.weight,self.v_proj.bias)
    q=q.reshape(1,batch,heads,dim).permute(1,2,0,3)
    k=k.reshape(length,batch,heads,dim).permute(1,2,0,3)
    v=v.reshape(length,batch,heads,dim).permute(1,2,0,3)
    weights=torch.softmax(torch.matmul(q,k.transpose(-1,-2))*(dim**-0.5),dim=-1)
    pooled=torch.matmul(weights,v).permute(2,0,1,3).reshape(1,batch,width)
    return F.linear(pooled.squeeze(0),self.c_proj.weight,self.c_proj.bias)

if hasattr(model.visual,'attnpool'):
    model.visual.attnpool.forward=types.MethodType(attn_pool_forward,model.visual.attnpool)
for block in model.transformer.resblocks:
    block.attention=types.MethodType(attention,block)

out=root/'float_models'; out.mkdir(exist_ok=True)
image_input=torch.zeros(1,3,224,224,dtype=torch.float32)
tokens=clip.tokenize(['a photo of a dog'])
class TextEncoder(torch.nn.Module):
    def __init__(self,owner): super().__init__(); self.owner=owner
    def forward(self,text): return self.owner.encode_text(text)
text_wrapper=TextEncoder(model).eval()
with torch.no_grad():
    image_ref=model.encode_image(image_input).numpy()
    text_ref=model.encode_text(tokens).numpy()
torch.onnx.export(model.visual,image_input,str(out/'clip_rn50_image.onnx'),input_names=['image'],output_names=['image_features'],opset_version=11,dynamo=False)
torch.onnx.export(text_wrapper,tokens,str(out/'clip_rn50_text.onnx'),input_names=['texts'],output_names=['text_features'],opset_version=11,dynamo=False)
for name in ('clip_rn50_image.onnx','clip_rn50_text.onnx'):
    onnx.checker.check_model(onnx.load(str(out/name)))
image_session=ort.InferenceSession(str(out/'clip_rn50_image.onnx'),providers=['CPUExecutionProvider'])
text_session=ort.InferenceSession(str(out/'clip_rn50_text.onnx'),providers=['CPUExecutionProvider'])
image_onnx=image_session.run(None,{'image':image_input.numpy()})[0]
text_onnx=text_session.run(None,{'texts':tokens.numpy()})[0]
print('image cosine',float(np.dot(image_ref.ravel(),image_onnx.ravel())/(np.linalg.norm(image_ref)*np.linalg.norm(image_onnx))),flush=True)
print('text cosine',float(np.mean(np.sum(text_ref*text_onnx,axis=1)/(np.linalg.norm(text_ref,axis=1)*np.linalg.norm(text_onnx,axis=1)))),flush=True)
print('exported',out/'clip_rn50_image.onnx',out/'clip_rn50_text.onnx',flush=True)
