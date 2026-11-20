# ADR-0007: The vehicle detector

- **Status:** Accepted
- **Date:** 2026-09-23

## Context

The telemetry does not report the traffic around the truck, so following a
car, stopping behind one and noticing one pull in all depend on seeing it.
Phase 5 starts from a detector pretrained on COCO. How well it does on the
game is the baseline a fine-tune on ETS2 frames would have to beat.

These constraints decided the choice more than accuracy tables did:

- **The licence.** The repository is MIT. The plan is to ship ONNX models
  with releases so a user does not need PyTorch. Weights under a copyleft
  licence cannot ship that way.
- **The GPU is the game's.** On the development machine an RTX 2060 renders
  the game at 80 FPS and is 45–85 % busy doing it. The detector gets what is
  left. The Phase 5 target is under 50 ms from frame to detections, and the
  control loop wants it well under that.
- **The windscreen is wide.** The part of the window that shows the road
  ahead is about 1590×450 pixels, which is 3.5:1. A square input either
  squashes it to a third of its width or spends two thirds of the input on
  padding. Distant cars are a few pixels either way.

## Options considered

All three were exported to ONNX and run with ONNX Runtime 1.30 on the CUDA
execution provider, on the same saved 1080p game frame, with the game running
on the same GPU:

| model | licence | params | input | median | p95 |
|---|---|---|---|---|---|
| RF-DETR nano | Apache-2.0 | ~30 M | 384×384 | 25.2 ms | 29.1 ms |
| RF-DETR nano | Apache-2.0 | ~30 M | 384×1152 | 82.5 ms | 88.7 ms |
| RF-DETR nano | Apache-2.0 | ~30 M | 448×1344 | 107.2 ms | 121.0 ms |
| D-FINE nano | Apache-2.0 | 3.8 M | 640×640 | 12.0 ms | 20.1 ms |
| D-FINE nano | Apache-2.0 | 3.8 M | 384×1152 | 14.3 ms | 21.9 ms |
| D-FINE nano | Apache-2.0 | 3.8 M | 448×1344 | 20.8 ms | 24.4 ms |

On the CPU, RF-DETR nano took 214 ms at 384×384.

### Option A — YOLO11 (Ultralytics)

The obvious default, and what the TODO first named. The code and weights are
AGPL-3.0, and that rules it out: an MIT project cannot ship its weights, and
a network service built on it would inherit the AGPL. It was not benchmarked,
because no result would change the answer.

### Option B — RF-DETR nano (Roboflow)

Apache-2.0, with a DINOv2 backbone, which transfers well to images unlike its
training data. Its own exporter handles non-square inputs. The backbone is
also the problem: at the wide input the windscreen needs, the transformer
costs 80–110 ms beside the game. That is two to four times the latency budget.

### Option C — D-FINE nano (USTC, via Hugging Face transformers)

Apache-2.0, a CNN backbone with a light transformer decoder, 3.8 M
parameters, 42.8 AP on COCO. It runs the wide input in 21 ms. Its output is
the usual DETR form (300 queries, normalised boxes, one logit per class, no
NMS). Fine-tuning code is available under the same licence.

## Decision

**D-FINE nano, exported at 448×1344, run with ONNX Runtime; the GPU when
there is one, the CPU otherwise.** It is the only permissively licensed
option that fits the latency budget at an input shaped like the windscreen.

The code reads the model through a small JSON description written beside the
ONNX file: input size, normalisation, class names, source, licence and a
hash. Nothing in `core/perception/detection.py` is specific to D-FINE beyond
the DETR output shape. Swapping in a fine-tuned model, or RF-DETR on a faster
GPU, means exporting it with `tools/export_detector.py`.

## Consequences

- **Pretrained on COCO, it is not good enough.** On a held-out drive it
  found 26 % of the vehicles, and 15 % of the near ones: it missed the rear
  of a dump truck 15 m ahead. Fine-tuned on the game's own frames, with labels
  from D-FINE xlarge, it found 75 % overall and 94 % of the near ones, still
  in 19 ms (docs/results/vehicle_detection.md). The model the stack runs is
  therefore the fine-tuned one. It is made from local recordings with
  `tools/pseudo_label.py` and `tools/train_detector.py`, and like any
  weights it is not in the repository.
- The labels come from a larger detector, not from people. That is what made
  a fine-tune possible in an evening, and it caps the student at the
  teacher's quality. Hand-checked labels for an evaluation set are the
  eventual answer.
- `onnxruntime-gpu` with pip's CUDA 13 and cuDNN is about 2.5 GB. It is the
  `perception` extra, with `perception-cpu` as the small alternative, and
  neither is needed to run the tests.
- `tools/train_detector.py` writes the same model.json as
  `tools/export_detector.py`, so the stack loads a fine-tuned model like the
  pretrained one.
- TensorRT with FP16 could cut the 19 ms further. It is a later optimisation.
  The engine would be built on the user's machine, and never shipped.
- Bigger pretrained models did not help. D-FINE small and medium
  (obj365→COCO) took 50 and 76 ms and still missed the distant vehicles the
  nano model missed. Only xlarge found them, at 180 ms, which is why it is
  the offline teacher and not the detector.
