import pytest

from src.utils.fps import FPSCounter


def test_fps_from_regular_ticks():
    counter = FPSCounter(window=10)
    for i in range(10):
        fps = counter.tick(now=i * 0.05)  # um frame a cada 50 ms
    assert fps == pytest.approx(20.0)


def test_fps_zero_before_two_frames():
    counter = FPSCounter()
    assert counter.tick(now=1.0) == 0.0
