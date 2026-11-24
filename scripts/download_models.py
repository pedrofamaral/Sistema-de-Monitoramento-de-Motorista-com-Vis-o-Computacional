"""Baixa os pesos pré-treinados usados pelo projeto para a pasta models/.

Uso: python scripts/download_models.py
"""
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config, project_path  # noqa: E402

FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)


def main() -> None:
    config = load_config()

    face_path = project_path(config["models"]["face_landmarker"])
    if face_path.exists():
        print(f"[ok] {face_path.name} já existe")
    else:
        face_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Baixando {face_path.name}...")
        urllib.request.urlretrieve(FACE_LANDMARKER_URL, face_path)
        print(f"[ok] {face_path} ({face_path.stat().st_size / 1e6:.1f} MB)")

    # O Ultralytics baixa os pesos oficiais sozinho quando o arquivo não existe.
    from ultralytics import YOLO

    yolo_path = project_path(config["models"]["yolo"])
    YOLO(str(yolo_path))
    print(f"[ok] {yolo_path} ({yolo_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
