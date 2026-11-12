# RDK X5 CLIP Photo Search: Verified Experiment Report

Date: 2026-11-02 (America/Toronto). Hardware: D-Robotics RDK X5 V1.0, 6.9 GiB RAM, Ubuntu 22.04.5. Toolchain container: OpenExplorer X5 CPU v1.2.8, `hb_mapper` 1.24.3, HBDK 3.49.15. Target architecture: `bayes-e`.

## Result

The selected image encoder is OpenAI CLIP RN50, quantized with 256 independent COCO 2017 calibration images. The text encoder remains the paired FP32 ONNX encoder on CPU.

Final artifact:

```text
D:\webinar\clip-photo-search\build\clip_rn50_coco256_default\clip_rn50_x5_coco256.bin
SHA-256 7261675b3931dc1ccd4bccde73284f0ae8d25f63f1f8f96d9cfbf6bd4ef2faa2
size    36,642,152 bytes (34.9 MiB)
```

The model was uploaded and executed on the X5. Its input is `float32`, NCHW, `1x3x224x224`; its output is `1x1024`. Preprocessing is the original CLIP pipeline: bicubic resize the shorter edge to 224, center crop, RGB, scale to [0,1], then normalize with CLIP mean and standard deviation.

## Dataset split

The original CLIP training corpus is OpenAI's unreleased WIT collection of approximately 400 million image-text pairs. This project does not claim access to WIT. For PTQ and evaluation it uses COCO 2017:

- 256 calibration images.
- 512 evaluation images.
- The two sets are disjoint.
- Split order is deterministic SHA-256 order over COCO image IDs.
- The evaluation query is one official caption per image and the target is its source image.

This setup is suitable for regression and comparison, but not for broad claims about production search accuracy. COCO captions can also correctly describe more than one image.

## Precision and retrieval

| Image encoder | Recall@1 | Recall@5 | Recall@10 |
|---|---:|---:|---:|
| RN50 FP32 | 51.8% | 85.4% | 93.4% |
| RN50 PTQ on X5 | 52.7% | 83.2% | 92.4% |

The 0.98 percentage-point Recall@1 increase is a small ranking change under five-fold cross-image competition; it is not evidence that quantization improves the underlying model. Recall@5 and Recall@10 declined slightly.

Feature agreement between FP32 and quantized X5 embeddings over 512 images:

- Mean cosine: 0.902
- 5th percentile: 0.851
- Minimum cosine: 0.677
- Top-1 image agreement: 70.5%

## Performance

- X5 image encoder forward median: 14.54 ms
- X5 image encoder forward p95: 14.78 ms
- Indexing including image read and preprocessing: 33.5 ms/image median
- RN50 text encoder on X5 CPU: approximately 0.34 s/query
- End-to-end text search in the 128-image local index: approximately 0.35-0.39 s

The performance figures are single-process, single-thread forward measurements on one board. They are not throughput claims under concurrency.

## Why RN50 replaced ViT-B/32

The image branch from OpenAI ViT-B/32 also compiled and ran, but its quantized embeddings were not usable:

| ViT variant | Recall@1 | Feature cosine | X5 forward median |
|---|---:|---:|---:|
| ViT-B/32, 32 calibration images | 7.6% | 0.435 | 125.2 ms |
| ViT-B/32, 256 calibration images | 8.6% | 0.459 | 124.2 ms |
| RN50, 256 calibration images | 52.7% | 0.902 | 14.54 ms |

The ViT report showed severe cumulative error around LayerNorm. A targeted int16 experiment used node names that disappeared after graph optimization, so its measured output was identical to the original pilot. The failed experiments are retained rather than hidden.

## Reproduction

Install or import the official X5 CPU toolchain image once. A local alias avoids duplicate Docker Hub lookups:

```powershell
docker tag 088921b996c1 x5-oe:v1.2.8
```

The tutorial's executable cells run from `D:\webinar\clip-photo-search`. The main command is:

```powershell
docker run --rm --network none --memory 12g `
  --mount "type=bind,source=D:\webinar\clip-photo-search,target=/workspace" `
  -w /workspace x5-oe:v1.2.8 `
  hb_mapper makertbin --config /workspace/x5_rn50_coco256.yaml --model-type onnx
```

Validation artifacts:

```text
results/coco2017/rn50_baseline_fp32.json
results/coco2017/rn50_coco256_comparison.json
clip-photo-search/hb_mapper_makertbin.log
clip-photo-search/build/clip_rn50_coco256_default/
```

## Current limitations

- The demo corpus is public COCO imagery, not a production NAS.
- A real NAS adapter will need credentials, share mounting policy, incremental indexing and deletion/re-index behavior.
- The text encoder is still FP32 on CPU because query frequency is low; quantizing it would require its own operator, precision and latency experiment.
- Chinese-language search has not been evaluated. The current model and queries are English.
- The benchmark uses one caption and one ground-truth image per query.
- Docker Display on this Windows machine currently duplicates identical image rows in `docker image ls`; `docker system df` and image inspection confirm there is one physical OE image layer set.
