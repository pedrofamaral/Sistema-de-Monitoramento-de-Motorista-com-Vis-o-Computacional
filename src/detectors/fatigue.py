"""Fadiga: PERCLOS e frequência de bocejos em janelas deslizantes.

Sonolência (eyes.py) é o perigo imediato: olhos fechados agora, por 2 s.
Fadiga é o cansaço acumulado, medido ao longo de minutos, e serve como aviso
antecipado.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass


class PerclosMeter:
    """PERCLOS: fração do tempo com os olhos fechados numa janela deslizante.

    Cada frame pesa pelo tempo que representa (não pela contagem de frames),
    então o valor não depende do FPS. Momentos sem rosto ficam fora da conta.
    """

    def __init__(self, window_seconds: float, max_frame_gap: float = 0.5):
        self.window_seconds = window_seconds
        self.max_frame_gap = max_frame_gap  # limita o peso de um frame após uma travada
        self._samples: deque[tuple[float, float, bool]] = deque()  # (t, duração, fechado)
        self._last_t: float | None = None

    def update(self, eyes_closed: bool | None, t: float) -> None:
        if self._last_t is not None and eyes_closed is not None:
            self._samples.append((t, min(t - self._last_t, self.max_frame_gap), eyes_closed))
        self._last_t = t
        while self._samples and self._samples[0][0] <= t - self.window_seconds:
            self._samples.popleft()

    @property
    def observed_seconds(self) -> float:
        """Tempo com rosto visível dentro da janela."""
        return sum(dt for _, dt, _ in self._samples)

    @property
    def value(self) -> float:
        total = self.observed_seconds
        closed = sum(dt for _, dt, is_closed in self._samples if is_closed)
        return closed / total if total > 0 else 0.0


@dataclass
class FatigueStatus:
    perclos: float = 0.0           # 0 a 1
    observed_seconds: float = 0.0  # dados disponíveis na janela do PERCLOS
    ready: bool = False            # já há dados suficientes para avaliar o PERCLOS
    yawns: int = 0                 # bocejos na janela
    fatigued: bool = False         # FADIGA


class FatigueMonitor:
    """FADIGA = PERCLOS acima do limiar OU muitos bocejos na janela.

    O PERCLOS só é avaliado depois de `min_observed_seconds` de rosto visível,
    para uma piscada no início não virar "30% de olhos fechados". Tem histerese
    para o estado não oscilar quando o valor fica perto do limiar.
    """

    def __init__(self, perclos_window: float, perclos_threshold: float,
                 perclos_hysteresis: float, min_observed_seconds: float,
                 yawn_window: float, yawn_count: int):
        self.perclos = PerclosMeter(perclos_window)
        self.perclos_threshold = perclos_threshold
        self.perclos_hysteresis = perclos_hysteresis
        self.min_observed_seconds = min_observed_seconds
        self.yawn_window = yawn_window
        self.yawn_count = yawn_count
        self._yawn_times: deque[float] = deque()
        self._perclos_high = False

    def update(self, eyes_closed: bool | None, yawn_started: bool, t: float) -> FatigueStatus:
        self.perclos.update(eyes_closed, t)
        if yawn_started:
            self._yawn_times.append(t)
        while self._yawn_times and self._yawn_times[0] <= t - self.yawn_window:
            self._yawn_times.popleft()

        perclos, observed = self.perclos.value, self.perclos.observed_seconds
        ready = observed >= self.min_observed_seconds
        if not ready:
            self._perclos_high = False
        else:
            limit = self.perclos_threshold - (self.perclos_hysteresis if self._perclos_high else 0.0)
            self._perclos_high = perclos >= limit

        yawns = len(self._yawn_times)
        return FatigueStatus(perclos=perclos, observed_seconds=observed, ready=ready, yawns=yawns,
                             fatigued=self._perclos_high or yawns >= self.yawn_count)
