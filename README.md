# Quantizing CLIP on RDK X5

An interactive webinar guide covering CLIP architecture, ONNX export, post-training quantization, calibration, the X5 toolchain, and board-side inference.

**Open the guide:** https://shockley6668.github.io/rdk-x5-clip-webinar/

## What's included

- Step-by-step attention, contrastive-training, and quantization animations.
- Export and calibration examples, annotated X5 configuration, and compilation diagnostics.
- Recorded RN50 / ViT comparisons, with evaluation scope and limitations.
- An eight-step board runtime walkthrough and common inference APIs.
- Saved floating-point photo-retrieval examples and official RDK / agent resources.

The public website runs in your browser. Copy code snippets to run them in your own environment. Live X5 inference requires your own board, model files, dependencies, and photo index; this site does not connect to the presenter's device.

The saved search examples are FP32 RN50 reference rankings among 512 COCO images. They are separate from the quantized X5 measurements in the [experiment report](report.md). Model weights and compiled binaries are not included.

## Run the guide locally

```sh
python -m http.server 8765
```

Open http://127.0.0.1:8765/ in your browser.

## References

- [OpenAI CLIP code](https://github.com/openai/CLIP) and [paper](https://arxiv.org/abs/2103.00020).
- [COCO dataset](https://cocodataset.org/).
- [D-Robotics documentation](https://developer.d-robotics.cc/rdk_doc_center/en/).
- [RDK Model Zoo](https://github.com/D-Robotics/rdk_model_zoo).

Third-party code, images, and referenced materials retain their respective terms. The bundled example scripts and reports document this webinar's experiments; consult the linked upstream projects for their full setup and licenses.
