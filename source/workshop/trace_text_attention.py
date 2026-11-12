"""Export a faithful last-layer CLIP text-attention trace for the webinar UI.

Runs the local OpenAI RN50 checkpoint in FP32 on CPU. It does not modify models.
Every displayed attention value and search score is derived from this forward pass.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'upstream-openai-clip'))
import clip
from clip.simple_tokenizer import SimpleTokenizer


def vector(x):
    return x.detach().float().cpu().numpy().tolist()


def main():
    torch.set_num_threads(4)
    model, _ = clip.load(str(ROOT / 'RN50.pt'), device='cpu', jit=False)
    model.eval().float()
    tokenizer = SimpleTokenizer()
    index = np.load(ROOT.parent / 'results/coco2017/rn50_baseline_fp32.npz')
    assets = ROOT / 'workshop/assets/attention'
    assets.mkdir(parents=True, exist_ok=True)
    result = {
        'model': 'OpenAI CLIP RN50', 'precision': 'FP32 CPU',
        'checkpoint_sha256': hashlib.sha256((ROOT / 'RN50.pt').read_bytes()).hexdigest(),
        'layer': 12, 'layers': 12, 'heads': 8, 'head_dim': 64,
        'text_width': 512, 'output_dim': 1024, 'context_length': 77,
        'image_index': '512 held-out COCO 2017 images; paired RN50 FP32 embeddings',
        'queries': [],
    }
    for query in ('a photo of a dog', 'a photo of a bicycle', 'a bowl of food'):
        tokens = clip.tokenize([query])
        eot = int(tokens.argmax(dim=-1)[0])
        captured = {}
        hook = model.transformer.resblocks[-1].register_forward_pre_hook(
            lambda _module, inputs: captured.update(x=inputs[0].detach().clone()))
        with torch.no_grad():
            reference = model.encode_text(tokens)
        hook.remove()
        with torch.no_grad():
            block = model.transformer.resblocks[-1]
            x = captured['x']
            q, k, v = F.linear(block.ln_1(x), block.attn.in_proj_weight,
                               block.attn.in_proj_bias).chunk(3, dim=-1)
            def split(t):
                return t.reshape(77, 1, 8, 64).permute(1, 2, 0, 3)
            q, k, v = map(split, (q, k, v))
            logits = q @ k.transpose(-1, -2) / 8.0
            logits = logits + block.attn_mask
            weights = logits.softmax(dim=-1)
            heads = weights @ v
            concat = heads.permute(2, 0, 1, 3).reshape(77, 1, 512)
            projected = F.linear(concat, block.attn.out_proj.weight, block.attn.out_proj.bias)
            residual = x + projected
            updated = residual + block.mlp(block.ln_2(residual))
            final = model.ln_final(updated.permute(1, 0, 2))[:, eot] @ model.text_projection
            error = float((final - reference).abs().max())
            assert error < 1e-4, error
            normalized = final / final.norm(dim=-1, keepdim=True)
            scores = index['image_vectors'] @ normalized.numpy().ravel()
            top = np.argsort(-scores)[:3]
            photos = []
            for i in top:
                name = str(index['names'][i])
                image = Image.open(ROOT.parent / 'datasets/coco2017_subset/evaluation' / name)
                image.thumbnail((360, 240))
                image.convert('RGB').save(assets / name, quality=85)
                photos.append({'name': name, 'url': '/assets/attention/' + name,
                               'score': float(scores[i])})
            data = {
                'query': query, 'tokens': [tokenizer.decode([int(t)]).strip()
                                           for t in tokens[0, :eot+1]],
                'token_ids': tokens[0, :eot+1].tolist(), 'eot_index': eot,
                'eot_input_preview': vector(x[eot, 0, :6]),
                'heads': [], 'concat_preview': vector(concat[eot, 0, :6]),
                'projected_preview': vector(projected[eot, 0, :6]),
                'residual_preview': vector(residual[eot, 0, :6]),
                'updated_preview': vector(updated[eot, 0, :6]),
                'text_feature_preview': vector(normalized[0, :6]),
                'max_abs_reconstruction_error': error, 'photos': photos,
            }
            for h in range(8):
                w = weights[0, h, eot, :eot+1]
                values = v[0, h, :eot+1]
                data['heads'].append({
                    'q_preview': vector(q[0, h, eot, :6]),
                    'k_preview': vector(k[0, h, :eot+1, :6]),
                    'v_preview': vector(values[:, :6]),
                    'scores': vector(logits[0, h, eot, :eot+1]),
                    'weights': vector(w), 'output_preview': vector(heads[0, h, eot, :6]),
                    'weight_sum': float(w.sum()),
                })
            result['queries'].append(data)
            print(query, 'max_abs_error', error, 'top_score', float(scores[top[0]]), flush=True)
    (assets / 'trace.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf8')
    print('Wrote', assets / 'trace.json', flush=True)


if __name__ == '__main__':
    main()
