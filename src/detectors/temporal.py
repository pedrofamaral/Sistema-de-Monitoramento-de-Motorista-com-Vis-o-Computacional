"""Regras temporais: um sinal só vira alerta depois de durar um tempo mínimo."""
from __future__ import annotations


class PersistenceFilter:
    """Confirma um sinal depois de ele ficar ativo por pelo menos `min_seconds`.

    Falhas mais curtas que `gap_tolerance` (ex.: o YOLO perder o celular por
    alguns frames) não zeram a contagem. Com gap_tolerance = 0, qualquer frame
    inativo zera. O tempo vem dos timestamps dos frames, então o resultado não
    depende do FPS.
    """

    def __init__(self, min_seconds: float, gap_tolerance: float = 0.0):
        self.min_seconds = min_seconds
        self.gap_tolerance = gap_tolerance
        self._start: float | None = None        # quando o sinal ficou ativo
        self._last_active: float | None = None  # última vez em que esteve ativo

    def update(self, active: bool, t: float) -> bool:
        """Registra o sinal no instante t (segundos) e diz se está confirmado."""
        if active:
            if self._start is None:
                self._start = t
            self._last_active = t
        elif self._start is not None and t - self._last_active > self.gap_tolerance:
            self.reset()
        return self.confirmed

    @property
    def duration(self) -> float:
        """Há quanto tempo o sinal está ativo (0 se não está)."""
        return 0.0 if self._start is None else self._last_active - self._start

    @property
    def confirmed(self) -> bool:
        return self._start is not None and self.duration >= self.min_seconds

    def reset(self) -> None:
        self._start = self._last_active = None
