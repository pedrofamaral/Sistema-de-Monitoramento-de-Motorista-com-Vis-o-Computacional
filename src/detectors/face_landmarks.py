"""Landmarks faciais com o MediaPipe Face Landmarker (API Tasks).

O Face Landmarker é o sucessor do Face Mesh: devolve os mesmos 478 pontos
(468 do rosto + 10 das íris), então os índices de landmarks são os mesmos.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

# Os 6 pontos de cada olho usados no Eye Aspect Ratio (p1..p6 de Soukupová e
# Čech, 2016): p1 e p4 são os cantos, p2/p3 a pálpebra de cima, p6/p5 a de baixo.
# "Direito" e "esquerdo" do ponto de vista do motorista.
RIGHT_EYE = (33, 160, 158, 133, 153, 144)
LEFT_EYE = (362, 385, 387, 263, 373, 380)

# Boca pelos lábios internos: cantos e três pares cima/baixo. Com os lábios
# internos o MAR fica perto de zero com a boca fechada (os externos somam a
# espessura dos lábios).
MOUTH_CORNERS = (78, 308)
MOUTH_PAIRS = ((82, 87), (13, 14), (312, 317))
MOUTH_CONTOUR = (78, 82, 13, 312, 308, 317, 14, 87)  # para desenhar


class FaceLandmarkDetector:
    def __init__(self, model_path: str | Path, min_detection_confidence: float = 0.5,
                 min_presence_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Modelo não encontrado: {model_path}. Rode: python scripts/download_models.py")
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,  # usa rastreamento entre frames
            num_faces=1,  # só o motorista
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._last_ts_ms = -1

    def detect(self, image_bgr: np.ndarray, timestamp: float) -> np.ndarray | None:
        """Retorna um array (478, 2) com as coordenadas em pixels, ou None sem rosto."""
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        # O modo VIDEO exige timestamps (ms) estritamente crescentes.
        ts_ms = max(int(timestamp * 1000), self._last_ts_ms + 1)
        self._last_ts_ms = ts_ms

        result = self._landmarker.detect_for_video(mp_image, ts_ms)
        if not result.face_landmarks:
            return None
        h, w = image_bgr.shape[:2]
        return np.array([(p.x * w, p.y * h) for p in result.face_landmarks[0]],
                        dtype=np.float32)

    def close(self) -> None:
        self._landmarker.close()
