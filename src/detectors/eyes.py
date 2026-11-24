"""Abertura dos olhos (EAR) e detecção de sonolência com critério temporal."""
from __future__ import annotations

import statistics
from dataclasses import dataclass

import numpy as np

from .face_landmarks import LEFT_EYE, RIGHT_EYE
from .temporal import PersistenceFilter


def eye_aspect_ratio(points: np.ndarray) -> float:
    """EAR de um olho a partir dos 6 pontos p1..p6 (Soukupová e Čech, 2016).

    EAR = (|p2 - p6| + |p3 - p5|) / (2 |p1 - p4|): altura média do olho
    dividida pela largura. Cai perto de zero quando o olho fecha.
    """
    p1, p2, p3, p4, p5, p6 = points
    width = np.linalg.norm(p1 - p4)
    if width == 0:
        return 0.0
    return float((np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) / (2.0 * width))


def average_ear(landmarks: np.ndarray) -> float:
    """Média do EAR dos dois olhos."""
    right = eye_aspect_ratio(landmarks[list(RIGHT_EYE)])
    left = eye_aspect_ratio(landmarks[list(LEFT_EYE)])
    return (right + left) / 2.0


class EARCalibrator:
    """Mede o EAR de olhos abertos do motorista e calcula o limiar dele.

    O EAR de olho aberto varia com o rosto e com o modelo de landmarks, então um
    limiar fixo serve mal a pessoas diferentes. A mediana ignora as piscadas
    que acontecerem durante a calibração.
    """

    def __init__(self, seconds: float, ratio: float):
        self.seconds = seconds
        self.ratio = ratio
        self._samples: list[float] = []
        self._start: float | None = None
        self._elapsed = 0.0

    def update(self, ear: float | None, t: float) -> float | None:
        """Acumula uma amostra; retorna o limiar quando a calibração termina."""
        if ear is None:
            return None  # sem rosto: só espera
        if self._start is None:
            self._start = t
        self._samples.append(ear)
        self._elapsed = t - self._start
        if self._elapsed >= self.seconds:
            return self.ratio * statistics.median(self._samples)
        return None

    @property
    def progress(self) -> float:
        """Fração concluída (0 a 1)."""
        return min(1.0, self._elapsed / self.seconds) if self.seconds > 0 else 1.0


@dataclass
class EyeStatus:
    ear: float | None   # EAR médio no frame (None sem rosto)
    eyes_closed: bool   # olhos fechados neste frame (já com histerese)
    closed_for: float   # segundos de olhos fechados contínuos
    drowsy: bool        # SONOLENCIA confirmada


class DrowsinessDetector:
    """Olhos fechados de forma contínua por mais que `closed_seconds` = sonolência.

    Histerese: o olho é considerado fechado com EAR < limiar, mas só volta a
    aberto com EAR >= limiar + histerese. Assim, um EAR oscilando perto do
    limiar não liga e desliga o estado a cada frame.
    """

    def __init__(self, ear_threshold: float, hysteresis: float,
                 closed_seconds: float, face_lost_grace: float):
        self.ear_threshold = ear_threshold
        self.hysteresis = hysteresis
        self.face_lost_grace = face_lost_grace
        self._closed = False
        self._last_face_t: float | None = None
        self._timer = PersistenceFilter(min_seconds=closed_seconds)

    def update(self, ear: float | None, t: float) -> EyeStatus:
        if ear is None:
            # Sem rosto: mantém o último estado por pouco tempo, porque a cabeça
            # pode cair ou virar ao cochilar; depois disso, zera.
            if self._last_face_t is None or t - self._last_face_t > self.face_lost_grace:
                self._closed = False
        else:
            self._last_face_t = t
            limit = self.ear_threshold + (self.hysteresis if self._closed else 0.0)
            self._closed = ear < limit

        drowsy = self._timer.update(self._closed, t)
        return EyeStatus(ear=ear, eyes_closed=self._closed,
                         closed_for=self._timer.duration, drowsy=drowsy)

    def reset(self) -> None:
        self._closed = False
        self._timer.reset()
