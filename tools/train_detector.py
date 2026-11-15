#!/usr/bin/env python3
"""Fine-tune the small detector on labelled game frames, and export it.

    uv run --no-project --python 3.11 \\
        --with torch --with torchvision --with transformers --with onnx \\
        --with opencv-python-headless --with scipy \\
        python tools/train_detector.py data/datasets/ets2-01/labels.json data/models/dfine-nano-ets2

Trains on the frames `tools/pseudo_label.py` labelled, cropped to the
windscreen exactly as `core.perception.detection.Detector` crops them, then
writes `model.onnx` and `model.json` the same way `tools/export_detector.py`
does -- so the stack loads the result like any other model.

The model keeps its COCO class head. The game's vehicles are labelled with
the COCO class they are (car, truck, bus, ...), so nothing the pretrained
model knows about the other classes has to be thrown away and relearnt.

Needs PyTorch with CUDA, and the GPU to itself: close the game first.
Recordings named with --hold-out are not trained on; evaluate on them with
tools/eval_detector.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = "ustc-community/dfine-nano-coco"
SPEC_FORMAT = "ets2ad-detector-1"  # core.perception.detection.SPEC_FORMAT

# The windscreen of core.perception.detection.DEFAULT_CAB_VIEW, as fractions.
WINDSCREEN = (330 / 1920, 250 / 1080, 1.0, 700 / 1080)

# Our kinds, in the order of core.perception.detection.Kind, as COCO names.
KIND_TO_COCO = ["car", "truck", "bus", "motorcycle", "bicycle", "person", "traffic light"]
COCO_ALIASES = {"motorcycle": ("motorcycle", "motorbike")}


def coco_index(model_labels: dict[int, str], name: str) -> int:
    for alias in COCO_ALIASES.get(name, (name,)):
        for index, label in model_labels.items():
            if label == alias:
                return int(index)
    raise KeyError(name)


def load(labels_path: Path, hold_out: set[str], floor: float):
    document = json.loads(labels_path.read_text(encoding="utf-8"))
    by_image: dict[int, list] = {}
    for a in document["annotations"]:
        if a["score"] >= floor:
            by_image.setdefault(a["image_id"], []).append(a)
    train, held = [], []
    for image in document["images"]:
        item = (image, by_image.get(image["id"], []))
        (held if image["recording"] in hold_out else train).append(item)
    return train, held


def crop_box(width: int, height: int) -> tuple[int, int, int, int]:
    left, top, right, bottom = WINDSCREEN
    return round(left * width), round(top * height), round(right * width), round(bottom * height)


def targets_in_crop(annotations, crop, coco_of_kind) -> tuple[np.ndarray, list[int]]:
    """Labels in frame pixels as the model's targets: normalised (cx, cy, w, h)
    over the crop, cut at its edges, and COCO class indices."""
    left, top, right, bottom = crop
    crop_w, crop_h = right - left, bottom - top
    boxes, classes = [], []
    for a in annotations:
        x, y, w, h = a["bbox"]
        x0, y0 = max(x - left, 0.0), max(y - top, 0.0)
        x1, y1 = min(x + w - left, crop_w), min(y + h - top, crop_h)
        if x1 - x0 < 1 or y1 - y0 < 1:
            continue
        boxes.append(
            [(x0 + x1) / 2 / crop_w, (y0 + y1) / 2 / crop_h, (x1 - x0) / crop_w, (y1 - y0) / crop_h]
        )
        classes.append(coco_of_kind[a["category_id"]])
    return np.array(boxes, dtype=np.float32).reshape(-1, 4), classes


def mirrored(boxes: np.ndarray) -> np.ndarray:
    """Normalised (cx, cy, w, h) boxes of an image flipped left to right."""
    flipped = boxes.copy()
    flipped[:, 0] = 1.0 - flipped[:, 0]
    return flipped


def sample(item, size, coco_of_kind, augment: bool, rng=random):
    """One training example: the windscreen crop as a tensor, and its targets."""
    import cv2
    import torch

    image_info, annotations = item
    image = cv2.imread(image_info["file_name"], cv2.IMREAD_COLOR)
    height, width = image.shape[:2]
    left, top, right, bottom = crop_box(width, height)
    crop = image[top:bottom, left:right]
    rgb = cv2.cvtColor(cv2.resize(crop, (size[1], size[0])), cv2.COLOR_BGR2RGB)
    pixels = rgb.astype(np.float32) / 255.0

    boxes, classes = targets_in_crop(annotations, (left, top, right, bottom), coco_of_kind)

    if augment:
        if rng.random() < 0.5:
            pixels = pixels[:, ::-1]
            boxes = mirrored(boxes)
        gain, bias = rng.uniform(0.6, 1.4), rng.uniform(-0.08, 0.08)
        pixels = np.clip(pixels * gain + bias, 0.0, 1.0)

    tensor = torch.from_numpy(np.ascontiguousarray(pixels.transpose(2, 0, 1)))
    target = {
        "class_labels": torch.tensor(classes, dtype=torch.long),
        "boxes": torch.from_numpy(boxes),
    }
    return tensor, target


class Frames:
    """The training examples, as a map-style dataset for a DataLoader.

    Decoding a 1080p JPEG takes longer than a training step, so several
    worker processes do it; each seeds `random` on its own.
    """

    def __init__(self, items, size, coco_of_kind) -> None:
        self.items, self.size, self.coco_of_kind = items, size, coco_of_kind

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        return sample(self.items[index], self.size, self.coco_of_kind, True)


def collate(batch):
    import torch

    return torch.stack([b[0] for b in batch]), [b[1] for b in batch]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "labels", type=Path, nargs="+", help="labels.json files from tools/pseudo_label.py"
    )
    parser.add_argument("output", type=Path, help="model directory to write")
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--hold-out", nargs="*", default=[], help="recording names not to train on")
    parser.add_argument("--floor", type=float, default=0.5, help="labels less sure are not used")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--size", type=int, nargs=2, default=(448, 1344), metavar=("H", "W"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=6, help="processes decoding images")
    args = parser.parse_args()

    import torch
    import transformers
    from huggingface_hub import model_info
    from transformers import AutoImageProcessor, AutoModelForObjectDetection

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    info = model_info(args.base)
    model = AutoModelForObjectDetection.from_pretrained(args.base, revision=info.sha).to(device)
    processor = AutoImageProcessor.from_pretrained(args.base, revision=info.sha)
    if getattr(processor, "do_normalize", False):
        raise SystemExit("this script feeds [0, 1] pixels; the base model wants normalisation")
    labels = {int(k): v for k, v in model.config.id2label.items()}
    coco_of_kind = [coco_index(labels, name) for name in KIND_TO_COCO]

    train, held = [], []
    for labels_path in args.labels:
        more_train, more_held = load(labels_path, set(args.hold_out), args.floor)
        train += more_train
        held += more_held
    if not train:
        raise SystemExit("nothing to train on")
    print(
        f"{len(train)} training images ({sum(len(a) for _, a in train)} boxes), "
        f"{len(held)} held out, on {device}"
    )

    backbone = [p for n, p in model.named_parameters() if "backbone" in n]
    rest = [p for n, p in model.named_parameters() if "backbone" not in n]
    optimiser = torch.optim.AdamW(
        [{"params": backbone, "lr": args.lr / 10}, {"params": rest, "lr": args.lr}],
        weight_decay=1e-4,
    )
    steps = args.epochs * (len(train) // args.batch)
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=[args.lr / 10, args.lr], total_steps=steps, pct_start=0.1
    )

    loader = torch.utils.data.DataLoader(
        Frames(train, args.size, coco_of_kind),
        batch_size=args.batch,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
        collate_fn=collate,
        generator=torch.Generator().manual_seed(args.seed),
    )

    started = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        # Batch norm stays as pretrained. Updating its statistics from batches
        # of four collapsed every score to about 0.03 after a full run; frozen,
        # scores rise as they should (docs/results/vehicle_detection.md).
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()
        losses = []
        for step, (pixels, batch_targets) in enumerate(loader):
            pixels = pixels.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in batch_targets]
            loss = model(pixel_values=pixels, labels=targets).loss
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimiser.step()
            schedule.step()
            losses.append(float(loss))
            if step % 100 == 0:
                print(f"  step {step}/{len(loader)}: loss {np.mean(losses[-100:]):.3f}", flush=True)
        minutes = (time.perf_counter() - started) / 60
        print(
            f"epoch {epoch + 1}/{args.epochs}: loss {np.mean(losses):.3f} ({minutes:.1f} min)",
            flush=True,
        )

    model.eval().cpu()

    class Outputs(torch.nn.Module):
        def __init__(self, inner: torch.nn.Module) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, pixels: torch.Tensor):
            out = self.inner(pixel_values=pixels)
            return out.pred_boxes, out.logits

    args.output.mkdir(parents=True, exist_ok=True)
    onnx_path = args.output / "model.onnx"
    height, width = args.size
    torch.onnx.export(
        Outputs(model),
        torch.zeros(1, 3, height, width),
        str(onnx_path),
        input_names=["input"],
        output_names=["boxes", "logits"],
        opset_version=17,
        dynamo=False,
    )
    labels_sha = [hashlib.sha256(p.read_bytes()).hexdigest() for p in args.labels]
    spec = {
        "format": SPEC_FORMAT,
        "source": (
            f"https://huggingface.co/{args.base} fine-tuned on "
            + ", ".join(str(p) for p in args.labels)
        ),
        "revision": info.sha,
        "license": (info.card_data or {}).get("license", ""),
        "input_size": [height, width],
        "normalize": None,
        "classes": [labels.get(i) for i in range(model.config.num_labels)],
        "sha256": hashlib.sha256(onnx_path.read_bytes()).hexdigest(),
        # Trained on the game's signal heads: the stack then reads a light's
        # lamps only inside one (core.perception.lights). A model fine-tuned
        # on vehicles alone keeps the class and never finds one.
        "signal_heads": any(
            KIND_TO_COCO[a["category_id"]] == "traffic light" for _, boxes in train for a in boxes
        ),
        "training": {
            "labels_sha256": labels_sha,
            "hold_out": args.hold_out,
            "floor": args.floor,
            "epochs": args.epochs,
            "batch": args.batch,
            "lr": args.lr,
            "seed": args.seed,
            "images": len(train),
        },
        "exported_with": {"torch": torch.__version__, "transformers": transformers.__version__},
    }
    (args.output / "model.json").write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    print(f"-> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
