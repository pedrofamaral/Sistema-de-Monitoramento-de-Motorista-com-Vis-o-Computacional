"""Roda o código de referência (AlertEdge) com o MediaPipe atual, sem alterar os arquivos dele.

A referência usa mp.solutions.face_mesh, API removida nas versões recentes do
MediaPipe. Este script coloca no lugar um adaptador com a mesma interface
(FaceMesh.process), implementado sobre o FaceLandmarker, que devolve os mesmos
478 landmarks, e então executa o main.py original.

Pré-requisitos:
  git clone https://github.com/AreebaAminn/Driver-Drowsiness-Distraction-AlertEdge referencia/AlertEdge
  pip install onnxruntime pygame
  python scripts/download_models.py

Uso:
  python scripts/run_reference.py               # q fecha a janela
  python scripts/run_reference.py --seconds 20  # fecha sozinho após 20 s
"""
import argparse
import os
import runpy
import sys
import time
import types
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = ROOT / "referencia" / "AlertEdge"
FACE_MODEL = ROOT / "models" / "face_landmarker.task"


class FaceMeshCompat:
    """Mesma interface do antigo mp.solutions.face_mesh.FaceMesh."""

    def __init__(self, max_num_faces=1, refine_landmarks=True,
                 min_detection_confidence=0.5, min_tracking_confidence=0.5, **_):
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(FACE_MODEL)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=max_num_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._start = time.perf_counter()
        self._last_ms = -1

    def process(self, rgb):
        ts_ms = max(int((time.perf_counter() - self._start) * 1000), self._last_ms + 1)
        self._last_ms = ts_ms
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, ts_ms)
        faces = [types.SimpleNamespace(landmark=lms) for lms in result.face_landmarks]
        return types.SimpleNamespace(multi_face_landmarks=faces or None)

    def close(self):
        self._landmarker.close()


def instrument_window(seconds):
    """Conta os frames exibidos (para medir FPS) e, se pedido, fecha após `seconds`."""
    stats = {"frames": 0, "start": None}
    original_imshow, original_wait_key = cv2.imshow, cv2.waitKey

    def imshow(*args):
        stats["start"] = stats["start"] or time.perf_counter()
        stats["frames"] += 1
        original_imshow(*args)

    def wait_key(delay=0):
        key = original_wait_key(delay)
        if seconds and stats["start"] and time.perf_counter() - stats["start"] > seconds:
            return ord("q")
        return key

    cv2.imshow, cv2.waitKey = imshow, wait_key
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--seconds", type=float, default=0, help="fecha sozinho após N segundos")
    args = parser.parse_args()

    if not (REFERENCE_DIR / "main.py").exists():
        sys.exit(f"Referência não encontrada em {REFERENCE_DIR}. Veja os pré-requisitos no topo do arquivo.")
    if not FACE_MODEL.exists():
        sys.exit("Modelo do rosto não encontrado. Rode: python scripts/download_models.py")

    mp.solutions = types.SimpleNamespace(face_mesh=types.SimpleNamespace(FaceMesh=FaceMeshCompat))
    stats = instrument_window(args.seconds)

    os.chdir(REFERENCE_DIR)  # o main.py usa caminhos relativos (models/yolov8n.onnx)
    sys.path.insert(0, str(REFERENCE_DIR))
    runpy.run_path("main.py", run_name="__main__")

    if stats["frames"] > 1:
        elapsed = time.perf_counter() - stats["start"]
        print(f"Referência: {stats['frames']} frames em {elapsed:.1f} s = {stats['frames'] / elapsed:.1f} FPS")


if __name__ == "__main__":
    main()
