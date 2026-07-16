import numpy as np
import pytest
import src.desktop.webcam as wc


def test_capture_encodes_injected_frame_to_jpeg():
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    frame[:, :, 1] = 200  # a green frame
    out = wc.capture_frame_jpeg(grabber=lambda idx: frame, index=0)
    assert isinstance(out, bytes) and out[:2] == b"\xff\xd8"  # JPEG SOI


def test_capture_raises_when_grabber_fails():
    def boom(idx):
        raise RuntimeError("no camera 0")
    with pytest.raises(RuntimeError):
        wc.capture_frame_jpeg(grabber=boom, index=0)
