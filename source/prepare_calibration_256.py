"""Create 256 disjoint COCO2017 calibration tensors using CLIP preprocessing."""
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image

root=Path(__file__).resolve().parent
project=root/'clip-photo-search'
dataset=root/'datasets/coco2017_subset'
sys.path.insert(0,str(project/'upstream-openai-clip'))
from clip.clip import _transform
out=project/'calibration_coco256'
out.mkdir(exist_ok=True)
transform=_transform(224)
items=json.loads((dataset/'split.json').read_text(encoding='utf8'))['calibration']
for i,item in enumerate(items,1):
    image=Image.open(dataset/'calibration'/item['file_name']).convert('RGB')
    tensor=transform(image).unsqueeze(0).numpy().astype(np.float32)
    assert tensor.shape==(1,3,224,224) and np.isfinite(tensor).all()
    dest=out/(Path(item['file_name']).stem+'.bin')
    tensor.tofile(dest)
    assert dest.stat().st_size==602112
    if i%50==0: print('Prepared',i,'/',len(items),flush=True)
print('Prepared',len(items),'CLIP-normalized calibration tensors:',out,flush=True)
