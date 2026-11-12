"""Inspect and run the deployed CLIP RN50 pair on RDK X5, without changing its index."""
import argparse
from pathlib import Path
import sys
import time

import cv2
import hbm_runtime
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "sample/runtime/python"))
from simple_tokenizer import SimpleTokenizer


def preprocess(path):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Cannot decode {path}")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    if height < width:
        new_height, new_width = 224, round(width * 224 / height)
    else:
        new_width, new_height = 224, round(height * 224 / width)
    rgb = cv2.resize(rgb, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    top, left = (new_height - 224) // 2, (new_width - 224) // 2
    rgb = rgb[top:top + 224, left:left + 224]
    x = rgb.astype(np.float32).transpose(2, 0, 1) / 255.0
    mean = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)[:, None, None]
    std = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)[:, None, None]
    return np.ascontiguousarray(((x - mean) / std)[None], dtype=np.float32)


def normalize(values):
    vector = np.asarray(values, np.float32).reshape(-1)
    norm = np.linalg.norm(vector)
    if vector.shape != (1024,) or not np.isfinite(vector).all() or norm == 0:
        raise ValueError("Expected a finite, nonzero 1024-dimensional feature")
    return vector / norm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, help="Defaults to the first indexed photo")
    parser.add_argument("--query", default="a photo of a dog")
    parser.add_argument("--index", type=Path, default=ROOT / "rn50-index.npz")
    args = parser.parse_args()

    # The saved index contains image embeddings already produced by this model.
    with np.load(args.index, allow_pickle=False) as saved:
        vectors = saved["vectors"]
        paths = saved["paths"]
        photos = Path(str(saved["root"]))
    photo = args.image or photos / str(paths[0])

    runtime = hbm_runtime.HB_HBMRuntime(str(ROOT / "clip_rn50_x5_coco256.bin"))
    model_name = runtime.model_names[0]
    input_name = runtime.input_names[model_name][0]
    output_name = runtime.output_names[model_name][0]
    print("MODEL:", model_name)
    print("INPUT:", input_name, runtime.input_shapes[model_name][input_name],
          runtime.input_dtypes[model_name][input_name])
    print("OUTPUT:", output_name, runtime.output_shapes[model_name][output_name],
          runtime.output_dtypes[model_name][output_name])

    x = preprocess(photo)
    print("PHOTO:", photo)
    print("PREPROCESSED:", x.shape, x.dtype, "contiguous:", x.flags.c_contiguous)
    started = time.perf_counter()
    outputs = runtime.run({model_name: {input_name: x}})
    elapsed = (time.perf_counter() - started) * 1000
    raw = outputs[model_name][output_name]
    image_vector = normalize(raw)
    print("RAW OUTPUT:", np.asarray(raw).shape, np.asarray(raw).dtype)
    print("IMAGE VECTOR:", image_vector.shape, "L2 norm:", np.linalg.norm(image_vector))
    print(f"ONE IMAGE CALL: {elapsed:.2f} ms (no warm-up; not a benchmark)")

    tokenizer = SimpleTokenizer()
    start = tokenizer.encoder["<|startoftext|>"]
    end = tokenizer.encoder["<|endoftext|>"]
    ids = [start] + tokenizer.encode(args.query) + [end]
    if len(ids) > 77:
        ids = ids[:76] + [end]
    tokens = np.zeros((1, 77), dtype=np.int32)
    tokens[0, :len(ids)] = ids
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    text_session = ort.InferenceSession(str(ROOT / "clip_rn50_text.onnx"),
                                       sess_options=options, providers=["CPUExecutionProvider"])
    text_vector = normalize(text_session.run(None, {"texts": tokens})[0])
    print("TOKENS:", tokens.shape, tokens.dtype)
    print("TEXT VECTOR:", text_vector.shape)
    print("SELECTED PHOTO SIMILARITY:", float(image_vector @ text_vector))

    if vectors.ndim != 2 or vectors.shape[1] != 1024 or len(paths) != len(vectors):
        raise ValueError("Index must contain one 1024-dimensional vector per photo")
    scores = vectors @ text_vector
    top_ids = np.argsort(-scores)[:5]
    print("INDEX:", vectors.shape, "QUERY:", args.query)
    for rank, idx in enumerate(top_ids, 1):
        print(f"{rank}. {paths[idx]}  similarity={float(scores[idx]):.4f}")


if __name__ == "__main__":
    main()
