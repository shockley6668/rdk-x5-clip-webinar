"""Export equivalent visual attention as primitive ops supported by ONNX opset 11."""
import sys
import types
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as F
from PIL import Image

root=Path(__file__).resolve().parent
sys.path.insert(0,str(root/'upstream-openai-clip'))
import clip

torch.set_num_threads(4)
model,preprocess=clip.load(str(root/'ViT-B-32.pt'),device='cpu',jit=False)
model.eval().float()
photos=sorted((root/'coco128/images/train2017').glob('*.jpg'))[:5]
inputs=[preprocess(Image.open(p).convert('RGB')).unsqueeze(0) for p in photos]
with torch.no_grad():
    refs=[model.encode_image(x).numpy() for x in inputs]

def primitive_attention(self,x):
    a=self.attn
    length,batch,width=x.shape
    heads=a.num_heads
    dim=width//heads
    q,k,v=F.linear(x,a.in_proj_weight,a.in_proj_bias).chunk(3,dim=-1)
    q=q.reshape(length,batch,heads,dim).permute(1,2,0,3)
    k=k.reshape(length,batch,heads,dim).permute(1,2,0,3)
    v=v.reshape(length,batch,heads,dim).permute(1,2,0,3)
    weights=torch.softmax(torch.matmul(q,k.transpose(-1,-2)) * (dim ** -0.5),dim=-1)
    result=torch.matmul(weights,v).permute(2,0,1,3).reshape(length,batch,width)
    return F.linear(result,a.out_proj.weight,a.out_proj.bias)

for block in model.visual.transformer.resblocks:
    if block.attn_mask is not None:
        raise RuntimeError('Unexpected vision attention mask')
    block.attention=types.MethodType(primitive_attention,block)
with torch.no_grad():
    for p,x,ref in zip(photos,inputs,refs):
        out=model.encode_image(x).numpy()
        print('PyTorch primitive',p.name,'max_abs',float(np.max(np.abs(out-ref))),flush=True)
    target=root/'float_models/clip_vit_b32_image_opset11.onnx'
    torch.onnx.export(model.visual,inputs[0],str(target),input_names=['image'],output_names=['image_features'],opset_version=11,dynamo=False)
onnx.checker.check_model(onnx.load(str(target)))
session=ort.InferenceSession(str(target),providers=['CPUExecutionProvider'])
for p,x,ref in zip(photos,inputs,refs):
    got=session.run(None,{'image':x.numpy()})[0]
    cos=float(np.dot(got.ravel(),ref.ravel())/(np.linalg.norm(got)*np.linalg.norm(ref)))
    print('ONNX opset11',p.name,'cosine',cos,'max_abs',float(np.max(np.abs(got-ref))),flush=True)
print('exported',target,flush=True)
