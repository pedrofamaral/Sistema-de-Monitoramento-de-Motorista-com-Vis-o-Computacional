"""Medição de FPS numa janela deslizante."""
from __future__ import annotations

import time
from collections import deque


class FPSCounter:
    """FPS médio dos últimos `window` frames (estável e sem atraso grande)."""

    def __init__(self, window: int = 30):
        self._times: deque[float] = deque(maxlen=window)

    def tick(self, now: float | None = None) -> float:
        """Registra um frame processado e retorna o FPS atual."""
        self._times.append(time.perf_counter() if now is None else now)
        return self.fps

    @property
    def fps(self) -> float:
        if len(self._times) < 2:
            return 0.0
        elapsed = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / elapsed if elapsed > 0 else 0.0
