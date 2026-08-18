"""shape_detect.detect() runs torchvision's pretrained Mask R-CNN via an
injectable model + categories list, mirroring src/face_swap.py's
analyzer=None/swapper=None pattern -- tests never need the real (170MB,
downloaded-on-first-use) model weights. See
docs/superpowers/specs/2026-08-15-shape-detection-design.md.
"""
import io

import numpy as np
from PIL import Image

from src import shape_detect


def _make_png(size=(4, 4), color=(200, 50, 50)):
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class _FakeModel:
    """Mirrors the real torchvision model's calling convention (called with
    a list of tensors, returns a list of dicts with boxes/labels/scores/
    masks) but returns plain numpy arrays -- detect()'s _to_numpy() helper
    accepts either, so fakes never need to construct real torch tensors."""

    def __init__(self, boxes, labels, scores, masks):
        self._out = {
            "boxes": np.array(boxes, dtype=np.float32),
            "labels": np.array(labels, dtype=np.int64),
            "scores": np.array(scores, dtype=np.float32),
            "masks": np.array(masks, dtype=np.float32),
        }

    def __call__(self, tensors):
        return [self._out]


_CATEGORIES = ["__background__", "person", "dog", "N/A"]


def test_detect_returns_labeled_detection_with_thresholded_mask():
    # 2x2 toy mask: top-right and bottom-left pixels are "on" (>= 0.5).
    model = _FakeModel(
        boxes=[[1, 2, 3, 4]], labels=[1], scores=[0.92],
        masks=[[[[0.1, 0.9], [0.8, 0.2]]]],
    )

    dets = shape_detect.detect(_make_png(size=(2, 2)), model=model, categories=_CATEGORIES)

    assert len(dets) == 1
    d = dets[0]
    assert d["label"] == "person"
    assert d["confidence"] == 0.92
    assert d["box"] == [1, 2, 3, 4]
    assert d["position"]  # non-empty; exact grid label is _position()'s own concern
    assert d["mask"].dtype == bool
    assert d["mask"].tolist() == [[False, True], [True, False]]


def test_detect_filters_below_confidence_threshold():
    model = _FakeModel(
        boxes=[[0, 0, 1, 1]], labels=[1], scores=[0.2],
        masks=[[[[0.9, 0.9], [0.9, 0.9]]]],
    )

    dets = shape_detect.detect(_make_png(size=(2, 2)), model=model, categories=_CATEGORIES, conf=0.4)

    assert dets == []


def test_detect_skips_background_and_na_labels():
    model = _FakeModel(
        boxes=[[0, 0, 1, 1], [0, 0, 1, 1]], labels=[0, 3], scores=[0.99, 0.99],
        masks=[[[[0.9, 0.9], [0.9, 0.9]]], [[[0.9, 0.9], [0.9, 0.9]]]],
    )

    dets = shape_detect.detect(_make_png(size=(2, 2)), model=model, categories=_CATEGORIES)

    assert dets == []


def test_detect_does_not_construct_real_model_when_injected(monkeypatch):
    """Injection must genuinely bypass real (170MB, network-fetching) model
    construction, not just the inference call."""
    def _fail_if_called():
        raise AssertionError("should not construct the real Mask R-CNN model")

    monkeypatch.setattr(shape_detect, "_get_model", _fail_if_called)
    monkeypatch.setattr(shape_detect, "_get_categories", _fail_if_called)

    model = _FakeModel(boxes=[], labels=[], scores=[], masks=np.empty((0, 1, 2, 2)))
    shape_detect.detect(_make_png(size=(2, 2)), model=model, categories=_CATEGORIES)


def test_detect_returns_empty_list_when_nothing_detected():
    model = _FakeModel(boxes=[], labels=[], scores=[], masks=np.empty((0, 1, 2, 2)))

    dets = shape_detect.detect(_make_png(size=(2, 2)), model=model, categories=_CATEGORIES)

    assert dets == []


def test_detect_numbers_detections_per_label():
    """Per the spec, detections are numbered PER LABEL (person #1, person
    #2, dog #1, ...) so a future swap_shape sub-project can disambiguate
    "swap the 2nd person" without a later breaking change to Detection's
    shape. Two people then a dog (in that order) must number the people
    1, 2 independently of the dog, which starts its own count at 1."""
    model = _FakeModel(
        boxes=[[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]],
        labels=[1, 1, 2], scores=[0.9, 0.9, 0.9],
        masks=[
            [[[0.9, 0.9], [0.9, 0.9]]],
            [[[0.9, 0.9], [0.9, 0.9]]],
            [[[0.9, 0.9], [0.9, 0.9]]],
        ],
    )

    dets = shape_detect.detect(_make_png(size=(2, 2)), model=model, categories=_CATEGORIES)

    assert [d["label"] for d in dets] == ["person", "person", "dog"]
    assert [d["index"] for d in dets] == [1, 2, 1]
