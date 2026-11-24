"""Detecção de celular com YOLO (Ultralytics), pesos pré-treinados no COCO."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ultralytics import YOLO


@dataclass
class Detection:
    box: tuple[int, int, int, int]  # x1, y1, x2, y2 em pixels
    confidence: float


class PhoneDetector:
    def __init__(self, model_path: str | Path, class_name: str = "cell phone",
                 confidence: float = 0.5, imgsz: int = 640):
        self.model = YOLO(str(model_path))
        # Procura a classe pelo nome, assim um modelo com fine-tuning (com outra
        # numeração de classes) funciona sem mudar o código.
        ids = [i for i, name in self.model.names.items() if name == class_name]
        if not ids:
            raise ValueError(f"Classe '{class_name}' não existe no modelo {model_path}")
        self.class_id = ids[0]
        self.confidence = confidence
        self.imgsz = imgsz

    def detect(self, image_bgr: np.ndarray) -> list[Detection]:
        result = self.model.predict(image_bgr, classes=[self.class_id], conf=self.confidence,
                                    imgsz=self.imgsz, verbose=False)[0]
        return [Detection(box=tuple(int(v) for v in xyxy), confidence=float(conf))
                for xyxy, conf in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist())]
