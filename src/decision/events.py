"""Eventos: transforma a lista de estados de cada frame em eventos com início e fim."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .engine import DriverState

# Para o EAR, o pior valor de um evento é o mínimo; para os outros, o máximo.
LOWER_IS_WORSE = {DriverState.DROWSY}


@dataclass
class Event:
    state: DriverState
    start: float           # timestamp do frame em que o alerta começou (s)
    started_at: datetime   # data e hora reais do início
    detail: str = ""       # descrição legível no início (vai para o CSV e para o Telegram)
    peak: float | None = None  # pior valor durante o evento (EAR mínimo, confiança/PERCLOS máximos)
    end: float | None = None

    @property
    def duration(self) -> float:
        return 0.0 if self.end is None else self.end - self.start


class EventTracker:
    def __init__(self):
        self._open: dict[DriverState, Event] = {}

    def update(self, states: list[DriverState], t: float,
               values: dict[DriverState, float | None] | None = None,
               details: dict[DriverState, str] | None = None) -> tuple[list[Event], list[Event]]:
        """Registra os estados do frame; retorna (eventos que começaram, que terminaram)."""
        values, details = values or {}, details or {}
        active = [state for state in states if state != DriverState.NORMAL]

        started = []
        for state in active:
            if state not in self._open:
                self._open[state] = Event(state, t, datetime.now(), details.get(state, ""))
                started.append(self._open[state])

        ended = []
        for state in [s for s in self._open if s not in active]:
            event = self._open.pop(state)
            event.end = t
            ended.append(event)

        for state, event in self._open.items():
            value = values.get(state)
            if value is None:
                continue
            if event.peak is None:
                event.peak = value
            else:
                event.peak = min(event.peak, value) if state in LOWER_IS_WORSE else max(event.peak, value)
        return started, ended

    def close(self, t: float) -> list[Event]:
        """Encerra os eventos abertos (fim do vídeo ou programa fechado)."""
        ended = list(self._open.values())
        for event in ended:
            event.end = t
        self._open.clear()
        return ended
