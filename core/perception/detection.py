"""Finding vehicles and people in the game window.

The telemetry says nothing about the traffic around the truck, so what is in
front of it has to be seen. A detector pretrained on COCO is the baseline:
cars, trucks and buses in ETS2 look enough like real ones for a model that
has never seen the game, and how well it does is what a fine-tune would be
measured against.

The model is an ONNX file with a small JSON beside it saying what it
expects -- input size, normalisation, what each class index means -- so the
code here knows nothing about any one architecture beyond the output shape
every DETR-style detector shares: a fixed set of queries, each with a
normalised (cx, cy, w, h) box and one logit per class, and no NMS to run.
`tools/export_detector.py` makes the pair.

Not all of the window is the road ahead. The cab has mirrors -- the real ones
on the doors and the game's picture-in-picture insets -- which show traffic
*behind*, mirrored; and the HUD draws over the corners. The model sees only
the windscreen (`CabView.windscreen`), and anything it finds that lies mostly
inside a mirror or the HUD is dropped (`CabView.covered`).
"""

from __future__ import annotations

import enum
import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "DEFAULT_CAB_VIEW",
    "CabView",
    "Detection",
    "Detector",
    "Kind",
    "ModelSpec",
    "Rect",
    "decode",
    "merge",
    "preprocess",
    "spans_view",
]

SPEC_FILE: Final = "model.json"
MODEL_FILE: Final = "model.onnx"
SPEC_FORMAT: Final = "ets2ad-detector-1"

DEFAULT_THRESHOLD: Final = 0.5

BIG_THRESHOLD: Final = 0.25
BIG_SHARE: Final = 0.26
"""A vehicle this much of the frame's width across is kept from this certainty:
near, and big, its score falls as it fills the windscreen. At night out of
Aberdeen a coach crossing ahead, given way to, scored 0.25 to 0.58 from 10 m:
under the threshold it was gone, the road was clear, and the truck drove into
its side. Over 11,782 views of the road this kept 26 boxes more; of 16 looked
at, all but a sign beside a bridge were vehicles, most beside the way."""

HEAD_THRESHOLD: Final = 0.25
"""Signal heads are kept from this certainty: a head only lets a lamp in it
be read (core.perception.lights), and approaching a red in Milan, the
head round the red lamp at 40 and 35 m scored 0.40 and 0.26 -- at the
vehicles' threshold the red was read at 14 m, not 42."""
HIDDEN_SHARE: Final = 0.5
"""A box with at least this share of its area in a mirror or the HUD is in it.

Not the box's centre: a car close alongside can have its centre on the
small door mirror and still be a car, most of it through the windscreen.
"""

NESTED_SHARE: Final = 0.85
"""A box this much inside another, larger one is part of the same object..."""

NESTED_MARGIN: Final = 0.1
"""...if the larger one is about as certain. Without this, a vague box over
half the windscreen at 0.35 swallowed the cars inside it at 0.8."""

WHOLE_VIEW_SHARE: Final = 0.75
"""A box wider than this share of what the model saw is not a vehicle.

Both the pretrained teacher and a student trained on its labels sometimes
box the whole dark windscreen, or a band of sky, as a car. Taken for real,
that is a car at the bumper and a hard brake on an empty road. A real car
this wide -- about 1190 pixels of a 1080p frame's windscreen -- would be
some two metres from the cab.
"""

DUPLICATE_IOU: Final = 0.7
"""Two boxes overlapping this much are one object seen as two classes.

One-to-one query matching keeps a DETR from boxing an object twice as the
*same* class, but a van can come out as a car and a truck from two queries.
"""


class Kind(enum.StrEnum):
    CAR = "car"
    TRUCK = "truck"
    BUS = "bus"
    MOTORCYCLE = "motorcycle"
    BICYCLE = "bicycle"
    PERSON = "person"
    TRAFFIC_LIGHT = "traffic light"
    """A signal head: where a light's lamps can be (core.perception.lights).
    Last, so a dataset's category numbers for the others stay as they were."""


# COCO names, and the spellings some exports use for them.
_KIND_OF_NAME: Final = {
    "car": Kind.CAR,
    "truck": Kind.TRUCK,
    "bus": Kind.BUS,
    "motorcycle": Kind.MOTORCYCLE,
    "motorbike": Kind.MOTORCYCLE,
    "bicycle": Kind.BICYCLE,
    "person": Kind.PERSON,
    "traffic light": Kind.TRAFFIC_LIGHT,
}


@dataclass(frozen=True, slots=True)
class Detection:
    kind: Kind
    score: float
    box: tuple[float, float, float, float]
    """(left, top, right, bottom) in pixels of the whole frame."""

    @property
    def centre(self) -> tuple[float, float]:
        left, top, right, bottom = self.box
        return (left + right) / 2.0, (top + bottom) / 2.0

    @property
    def ground_point(self) -> tuple[float, float]:
        """The middle of the box's bottom edge: where the object meets the road."""
        left, _, right, bottom = self.box
        return (left + right) / 2.0, bottom


@dataclass(frozen=True, slots=True)
class Rect:
    """A region of the frame as fractions of its width and height.

    Fractions rather than pixels so one layout serves any window size with
    the same aspect ratio.
    """

    left: float
    top: float
    right: float
    bottom: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.left < self.right <= 1.0 and 0.0 <= self.top < self.bottom <= 1.0):
            raise ValueError(f"{self} is not a region inside the frame")

    def pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        return (
            round(self.left * width),
            round(self.top * height),
            round(self.right * width),
            round(self.bottom * height),
        )

    def contains(self, x: float, y: float, width: int, height: int) -> bool:
        return (
            self.left * width <= x < self.right * width
            and self.top * height <= y < self.bottom * height
        )


@dataclass(frozen=True, slots=True)
class CabView:
    windscreen: Rect
    """What the model is shown: the road ahead through the windscreen."""

    covered: tuple[Rect, ...] = ()
    """Mirrors and HUD: a detection mostly inside one of these is not ahead."""

    def is_covered(self, x: float, y: float, width: int, height: int) -> bool:
        return any(rect.contains(x, y, width, height) for rect in self.covered)

    def hides(self, box: tuple[float, float, float, float], width: int, height: int) -> bool:
        """Whether a box lies mostly in a mirror or the HUD (`HIDDEN_SHARE`)."""
        area = (box[2] - box[0]) * (box[3] - box[1])
        if area <= 0:
            return False
        return any(
            _overlap(box, rect.pixels(width, height)) >= HIDDEN_SHARE * area
            for rect in self.covered
        )


DEFAULT_CAB_VIEW: Final = CabView(
    # Measured on a 1920x1080 frame from the default interior camera of a
    # Mercedes Actros: the windscreen and the road through the left door
    # window, down to the dashboard.
    windscreen=Rect(330 / 1920, 250 / 1080, 1.0, 700 / 1080),
    covered=(
        Rect(0 / 1920, 80 / 1080, 262 / 1920, 405 / 1080),  # top-left mirror inset
        Rect(1655 / 1920, 80 / 1080, 1.0, 405 / 1080),  # top-right mirror inset
        Rect(0 / 1920, 695 / 1080, 250 / 1920, 950 / 1080),  # bottom-left mirror inset
        Rect(0 / 1920, 380 / 1080, 330 / 1920, 950 / 1080),  # the left door mirror
        Rect(1430 / 1920, 570 / 1080, 1550 / 1920, 700 / 1080),  # the right door mirror
        # HUD: money, navigation, clock -- and, with a job on, the cargo and
        # destination above them, from 560.
        Rect(1560 / 1920, 560 / 1080, 1.0, 1.0),
        Rect(0 / 1920, 890 / 1080, 205 / 1920, 1.0),  # HUD: speedometer
    ),
)


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """What a model expects and what its outputs mean."""

    input_size: tuple[int, int]
    """(height, width) of the model's input."""

    classes: tuple[str | None, ...]
    """The name for each class index of the logits; None for unused indices."""

    mean: tuple[float, float, float] | None = None
    std: tuple[float, float, float] | None = None
    """Per-channel (RGB) normalisation after scaling to [0, 1]; None for none."""

    source: str = ""
    licence: str = ""
    sha256: str = ""
    signal_heads: bool = False
    """Whether it was trained to find the game's signal heads
    (`Kind.TRAFFIC_LIGHT`); one fine-tuned on vehicles alone keeps the class
    and never finds one."""

    @classmethod
    def from_json(cls, document: dict) -> ModelSpec:
        if document.get("format") != SPEC_FORMAT:
            raise ValueError(f"not a detector spec ({document.get('format')!r})")
        normalise = document.get("normalize")
        return cls(
            input_size=tuple(document["input_size"]),
            classes=tuple(document["classes"]),
            mean=tuple(normalise["mean"]) if normalise else None,
            std=tuple(normalise["std"]) if normalise else None,
            source=document.get("source", ""),
            licence=document.get("license", ""),
            sha256=document.get("sha256", ""),
            signal_heads=bool(document.get("signal_heads", False)),
        )

    def kinds(self) -> NDArray[np.int16]:
        """For each class index, the index of its `Kind` in `list(Kind)`, or -1."""
        order = list(Kind)
        return np.array(
            [order.index(_KIND_OF_NAME[n]) if n in _KIND_OF_NAME else -1 for n in self.classes],
            dtype=np.int16,
        )


def preprocess(image: NDArray[np.uint8], spec: ModelSpec) -> NDArray[np.float32]:
    """A BGR crop as the model's (1, 3, height, width) float input.

    Resized without keeping the aspect ratio: DETR-style detectors are
    trained on stretched images, and letterboxing a wide windscreen would
    spend most of the input on padding.
    """
    import cv2

    height, width = spec.input_size
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
    rgb *= 1.0 / 255.0
    if spec.mean is not None and spec.std is not None:
        rgb -= np.asarray(spec.mean, dtype=np.float32)
        rgb /= np.asarray(spec.std, dtype=np.float32)
    return np.ascontiguousarray(rgb.transpose(2, 0, 1)[np.newaxis])


def kept_canvas(
    crop: NDArray[np.uint8], input_size: tuple[int, int]
) -> tuple[NDArray[np.uint8], int]:
    """`crop` scaled to the model's input height with its shape kept, at the
    left of a grey canvas of the model's input size; and the width it takes.
    A crop too wide for the canvas is squeezed to its width."""
    import cv2

    height, width = input_size
    used = min(width, max(1, round(crop.shape[1] * height / max(crop.shape[0], 1))))
    canvas = np.full((height, width, 3), 114, np.uint8)
    canvas[:, :used] = cv2.resize(crop, (used, height), interpolation=cv2.INTER_LINEAR)
    return canvas, used


def decode(
    boxes: NDArray[np.floating],
    logits: NDArray[np.floating],
    kinds: NDArray[np.int16],
    crop: tuple[int, int, int, int],
    threshold: float = DEFAULT_THRESHOLD,
) -> list[Detection]:
    """Detections from one image's raw outputs, in frame pixels.

    `boxes` is (queries, 4) normalised (cx, cy, w, h) over the crop, `logits`
    (queries, classes). Each query is taken as its most likely class; queries
    whose class is not a `Kind` or whose score is under `threshold` are
    dropped, and of two boxes on one object only the more certain is kept.
    `crop` is where the crop was in the frame: (left, top, right, bottom).
    """
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    logits = np.asarray(logits, dtype=np.float64)
    logits = logits.reshape(len(boxes), logits.shape[-1])
    if logits.shape[1] != len(kinds):
        raise ValueError(f"{logits.shape[1]} class logits for a model with {len(kinds)} classes")
    best = logits.argmax(axis=1)
    scores = 1.0 / (1.0 + np.exp(-logits[np.arange(len(best)), best]))
    kind = kinds[best]
    keep = np.flatnonzero((kind >= 0) & (scores >= threshold))
    if keep.size == 0:
        return []
    keep = keep[np.argsort(-scores[keep], kind="stable")]

    left, top, right, bottom = crop
    width, height = right - left, bottom - top
    cx, cy, w, h = boxes[keep].T
    corners = np.column_stack(
        [
            left + (cx - w / 2) * width,
            top + (cy - h / 2) * height,
            left + (cx + w / 2) * width,
            top + (cy + h / 2) * height,
        ]
    )
    corners[:, 0::2] = corners[:, 0::2].clip(left, right)
    corners[:, 1::2] = corners[:, 1::2].clip(top, bottom)

    order = list(Kind)
    detections: list[Detection] = []
    kept_boxes: list[NDArray[np.float64]] = []
    for row, query in enumerate(keep):
        box = corners[row]
        if spans_view(tuple(box), crop):
            continue
        if any(_iou(box, other) > DUPLICATE_IOU for other in kept_boxes):
            continue
        kept_boxes.append(box)
        detections.append(
            Detection(order[kind[query]], float(scores[query]), tuple(float(v) for v in box))
        )
    return detections


def spans_view(box: Sequence[float], crop: Sequence[float]) -> bool:
    """Whether a box is wider than `WHOLE_VIEW_SHARE` of the crop it was found in."""
    return box[2] - box[0] > WHOLE_VIEW_SHARE * (crop[2] - crop[0])


def merge(*passes: Sequence[Detection], overlap: float = 0.5) -> list[Detection]:
    """Detections from several looks at one frame, each object once.

    A wide look and a zoomed one both find the nearer vehicles; of boxes
    overlapping by more than `overlap` (IoU), the most certain is kept,
    whatever the kind. A box nested inside a larger one (`NESTED_SHARE`) that
    is about as certain (`NESTED_MARGIN`) is taken for part of it -- the roof
    of a van boxed as well as the van -- and only the larger is kept.
    """
    ranked = sorted((d for found in passes for d in found), key=lambda d: -d.score)
    kept: list[Detection] = []
    for candidate in ranked:
        box = np.asarray(candidate.box)
        if all(_iou(box, np.asarray(other.box)) <= overlap for other in kept):
            kept.append(candidate)
    return [d for d in kept if not any(_absorbs(other, d) for other in kept)]


def _absorbs(outer: Detection, inner: Detection) -> bool:
    return (
        outer is not inner
        and outer.score >= inner.score - NESTED_MARGIN
        and _nested(inner.box, outer.box)
    )


def _area(box: Sequence[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _overlap(a: Sequence[float], b: Sequence[float]) -> float:
    across = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    down = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return across * down


def _nested(inner: Sequence[float], outer: Sequence[float]) -> bool:
    """Whether `inner` is the smaller box and lies almost wholly in `outer`."""
    size = _area(inner)
    return 0 < size < _area(outer) and _overlap(inner, outer) >= NESTED_SHARE * size


def _iou(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    across = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    down = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    overlap = across * down
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - overlap
    return overlap / union if union > 0 else 0.0


Run = Callable[[NDArray[np.float32]], tuple[NDArray[np.floating], NDArray[np.floating]]]
"""A model: input tensor in, (boxes, logits) for the one image out."""


class Detector:
    """Vehicles and people ahead, from a frame of the game window."""

    def __init__(
        self,
        run: Run,
        spec: ModelSpec,
        view: CabView = DEFAULT_CAB_VIEW,
        threshold: float = DEFAULT_THRESHOLD,
        provider: str = "",
    ) -> None:
        self._run = run
        self.spec = spec
        self.view = view
        self.threshold = threshold
        self.provider = provider
        """The ONNX Runtime execution provider the model runs on, if known."""
        self._kinds = spec.kinds()

    @classmethod
    def open(
        cls,
        directory: str | Path,
        view: CabView = DEFAULT_CAB_VIEW,
        threshold: float = DEFAULT_THRESHOLD,
        providers: Sequence[str] = ("CUDAExecutionProvider", "CPUExecutionProvider"),
    ) -> Detector:
        """A detector from a model directory made by tools/export_detector.py.

        Runs on the first of `providers` ONNX Runtime can use here -- the GPU
        when onnxruntime-gpu and its CUDA libraries are installed, the CPU
        otherwise. `provider` says which it got.
        """
        directory = Path(directory)
        spec = ModelSpec.from_json(json.loads((directory / SPEC_FILE).read_text("utf-8")))
        model = directory / MODEL_FILE
        if spec.sha256 and _sha256(model) != spec.sha256:
            raise ValueError(f"{model} is not the model {directory / SPEC_FILE} describes")

        import onnxruntime as ort

        # The pip CUDA and cuDNN wheels are not on the loader's path; this
        # loads them from where pip put them. Absent in CPU-only builds.
        if hasattr(ort, "preload_dlls") and "CUDAExecutionProvider" in providers:
            ort.preload_dlls()
        available = set(ort.get_available_providers())
        session = ort.InferenceSession(
            str(model), providers=[p for p in providers if p in available] or None
        )
        name = session.get_inputs()[0].name

        def run(tensor: NDArray[np.float32]):
            boxes, logits = session.run(None, {name: tensor})
            return boxes[0], logits[0]

        # The first run on a GPU sets up kernels and takes hundreds of
        # milliseconds; pay it here, not on the first frame that matters.
        run(np.zeros((1, 3, *spec.input_size), dtype=np.float32))
        return cls(run, spec, view, threshold, provider=session.get_providers()[0])

    def detect(self, image: NDArray[np.uint8]) -> list[Detection]:
        """What is ahead in one BGR frame of the whole game window, most certain first."""
        height, width = image.shape[:2]
        return [
            found
            for found in self._detect_in(image, self.view.windscreen)
            if not self.view.hides(found.box, width, height)
        ]

    def nearest_in_mirror(self, image: NDArray[np.uint8], side: int) -> float:
        """How near the nearest vehicle in the mirror on `side` is -- +1 the
        right, -1 the left (`MIRRORS`) -- as its box's height, a share of the
        mirror's; 0.0 for none. Not one centred on the truck's own trailer
        (`OWN_RIG`)."""
        region = MIRRORS[side]
        left, top, right, bottom = region.pixels(image.shape[1], image.shape[0])
        width, height = max(right - left, 1), max(bottom - top, 1)
        low, high = OWN_RIG[side]
        nearest = 0.0
        for found in self._detect_in(image, region):
            if found.kind is Kind.TRAFFIC_LIGHT:
                continue
            middle = ((found.box[0] + found.box[2]) / 2.0 - left) / width
            if low <= middle <= high:
                continue
            nearest = max(nearest, (found.box[3] - found.box[1]) / height)
        return nearest

    def far_in_mirror(self, image: NDArray[np.uint8], side: int) -> float:
        """How near the nearest vehicle in the mirror on `side` is, as
        `nearest_in_mirror` says, but found in the mirror as it looks: its
        crop scaled to the model's height, its shape kept, beside grey
        (`kept_canvas`). Stretched five times its width to the model's wide
        input, a car far back in the mirror was not found until 0.7 s before
        it hit the truck; kept, 1.8 s. Boxes taller than `MIRROR_FAR_MAX`
        are left out: kept, the mirror by night shows boxes on the glow of
        tail lights and on signs."""
        region = MIRRORS[side]
        left, top, right, bottom = region.pixels(image.shape[1], image.shape[0])
        canvas, used = kept_canvas(image[top:bottom, left:right], self.spec.input_size)
        height, width = self.spec.input_size
        boxes, logits = self._run(preprocess(canvas, self.spec))
        low, high = OWN_RIG[side]
        nearest = 0.0
        for found in decode(boxes, logits, self._kinds, (0, 0, width, height), self.threshold):
            if found.kind is Kind.TRAFFIC_LIGHT:
                continue
            middle = (found.box[0] + found.box[2]) / 2.0 / used
            share = (found.box[3] - found.box[1]) / height
            if middle > 1.0 or low <= middle <= high or share > MIRROR_FAR_MAX:
                continue
            nearest = max(nearest, share)
        return nearest

    def _detect_in(self, image: NDArray[np.uint8], region: Rect) -> list[Detection]:
        height, width = image.shape[:2]
        crop = region.pixels(width, height)
        left, top, right, bottom = crop
        boxes, logits = self._run(preprocess(image[top:bottom, left:right], self.spec))
        low = min(self.threshold, HEAD_THRESHOLD, BIG_THRESHOLD)
        found = decode(boxes, logits, self._kinds, crop, low)

        def wanted(d: Detection) -> float:
            if d.kind is Kind.TRAFFIC_LIGHT:
                return HEAD_THRESHOLD
            big = d.box[2] - d.box[0] >= BIG_SHARE * width
            return min(self.threshold, BIG_THRESHOLD) if big else self.threshold

        return [d for d in found if d.score >= wanted(d)]


MIRRORS: Final = {
    1: Rect(1655 / 1920, 80 / 1080, 1.0, 405 / 1080),  # the HUD's right mirror
    -1: Rect(0 / 1920, 80 / 1080, 262 / 1920, 405 / 1080),  # and its left
}
"""The game's mirror views at the top of the frame, by side: +1 the lane on
the right, behind; -1 the lane on the left (core.control.lane_change)."""

MIRROR_FAR_MAX: Final = 0.35
"""Of a mirror's height, the tallest box `Detector.far_in_mirror` counts."""

OWN_RIG: Final = {1: (0.0, 0.33), -1: (0.60, 1.0)}
"""Of each mirror's width, from its left, the share the truck's own trailer
fills with the rig straight (measured on the 8x4 with a box trailer, Bern,
6 October): a vehicle centred there is the trailer -- by night its marker
lamps, which the detector takes for a car's."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
