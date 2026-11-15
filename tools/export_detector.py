#!/usr/bin/env python3
"""Export a pretrained detector to ONNX, with the description the stack reads.

    uv run --no-project --python 3.11 \\
        --index https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match \\
        --with torch --with torchvision --with transformers --with onnx \\
        python tools/export_detector.py data/models/dfine-nano

Writes `model.onnx` and `model.json` into the directory. Needs PyTorch and
Hugging Face transformers, which the stack itself does not: it runs the ONNX
file with ONNX Runtime. So this runs in a throwaway environment, as above,
and the weights never enter the repository.

The default model is D-FINE nano pretrained on COCO (Apache-2.0), exported
for a wide input -- the windscreen is three times wider than it is tall. Why
this one, and not YOLO or RF-DETR, is in docs/adr/0007-vehicle-detector.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

DEFAULT_MODEL = "ustc-community/dfine-nano-coco"
SPEC_FORMAT = "ets2ad-detector-1"  # core.perception.detection.SPEC_FORMAT


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("output", type=Path, help="directory to write model.onnx and model.json")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model id")
    parser.add_argument("--revision", default=None, help="model revision; default the latest")
    parser.add_argument(
        "--size", type=int, nargs=2, default=(448, 1344), metavar=("HEIGHT", "WIDTH")
    )
    args = parser.parse_args()

    import torch
    import transformers
    from huggingface_hub import model_info
    from transformers import AutoImageProcessor, AutoModelForObjectDetection

    info = model_info(args.model, revision=args.revision)
    licence = (info.card_data or {}).get("license", "")
    processor = AutoImageProcessor.from_pretrained(args.model, revision=info.sha)
    model = AutoModelForObjectDetection.from_pretrained(args.model, revision=info.sha).eval()

    class Outputs(torch.nn.Module):
        """The two tensors every DETR-style detector ends in, as plain outputs."""

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

    labels = model.config.id2label
    classes = [labels.get(i) for i in range(model.config.num_labels)]
    normalize = (
        {"mean": list(processor.image_mean), "std": list(processor.image_std)}
        if getattr(processor, "do_normalize", False)
        else None
    )
    spec = {
        "format": SPEC_FORMAT,
        "source": f"https://huggingface.co/{args.model}",
        "revision": info.sha,
        "license": licence,
        "input_size": [height, width],
        "normalize": normalize,
        "classes": classes,
        "sha256": hashlib.sha256(onnx_path.read_bytes()).hexdigest(),
        "exported_with": {"torch": torch.__version__, "transformers": transformers.__version__},
    }
    (args.output / "model.json").write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    size_mb = onnx_path.stat().st_size / 1e6
    print(f"{args.model}@{info.sha[:10]} ({licence}), {height}x{width}, {size_mb:.1f} MB")
    print(f"-> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
