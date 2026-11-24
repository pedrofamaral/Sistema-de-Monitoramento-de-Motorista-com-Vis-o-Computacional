import cv2
import numpy as np
import pytest

from src.capture import VideoSource


@pytest.fixture
def video_file(tmp_path):
    """Vídeo sintético de 20 frames a 10 FPS (2 s), cada frame com um tom de cinza."""
    path = tmp_path / "test.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48))
    for i in range(20):
        writer.write(np.full((48, 64, 3), i * 10, dtype=np.uint8))
    writer.release()
    return path


def test_reads_all_frames_with_video_timestamps(video_file):
    with VideoSource(str(video_file)) as video:
        frames = list(video)
    assert len(frames) == 20
    assert [f.index for f in frames] == list(range(20))
    # Timestamp vem da posição no vídeo (frame i a 10 FPS = i * 0,1 s),
    # não da velocidade de processamento.
    assert [f.timestamp for f in frames] == pytest.approx([i * 0.1 for i in range(20)], abs=1e-3)


def test_max_fps_skips_frames_by_timestamp(tmp_path):
    path = tmp_path / "30fps.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (64, 48))
    for i in range(60):  # 2 s a 30 FPS
        writer.write(np.full((48, 64, 3), i * 4, dtype=np.uint8))
    writer.release()

    with VideoSource(str(path), max_fps=15) as video:
        frames = list(video)
        skipped = video.skipped
    assert len(frames) == 30 and skipped == 30          # 1 a cada 2 frames
    assert [f.index for f in frames[:3]] == [0, 2, 4]   # índice continua sendo o do vídeo
    assert frames[1].timestamp == pytest.approx(2 / 30, abs=1e-3)


def test_mirror_flips_horizontally(tmp_path):
    path = tmp_path / "half.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48))
    img = np.zeros((48, 64, 3), dtype=np.uint8)
    img[:, :32] = 255  # metade esquerda branca
    writer.write(img)
    writer.release()

    with VideoSource(str(path), mirror=True) as video:
        frame = video.read()
    assert frame.image[:, :24].mean() < 30    # esquerda ficou escura
    assert frame.image[:, -24:].mean() > 225  # direita ficou clara


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        VideoSource("nao_existe.mp4")
