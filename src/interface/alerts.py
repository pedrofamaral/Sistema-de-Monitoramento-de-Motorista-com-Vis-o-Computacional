"""Alerta sonoro em uma thread separada, para não travar o vídeo.

SONOLENCIA: bipe contínuo enquanto durar, para acordar o motorista.
CELULAR DETECTADO: dois bipes curtos no início. FADIGA: um bipe grave no início.
"""
from __future__ import annotations

import queue
import sys
import threading
import time

from ..decision.engine import DriverState
from ..decision.events import Event


def _beep(frequency: int, duration_ms: int) -> None:
    if sys.platform == "win32":
        import winsound
        winsound.Beep(frequency, duration_ms)
    else:  # sem winsound: usa o sino do terminal
        sys.stdout.write("\a")
        sys.stdout.flush()
        time.sleep(duration_ms / 1000)


class AlertSound:
    def __init__(self):
        self._alarm = threading.Event()   # SONOLENCIA ativa
        self._queue: queue.Queue[DriverState] = queue.Queue()
        threading.Thread(target=self._run, daemon=True).start()

    def update(self, states: list[DriverState], started: list[Event]) -> None:
        if DriverState.DROWSY in states:
            self._alarm.set()
        else:
            self._alarm.clear()
        for event in started:
            if event.state != DriverState.DROWSY:
                self._queue.put(event.state)

    def close(self) -> None:
        self._alarm.clear()

    def _run(self) -> None:
        while True:
            if self._alarm.is_set():
                _beep(1000, 400)
                time.sleep(0.2)
                continue
            try:
                state = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if state == DriverState.PHONE:
                _beep(1500, 150)
                time.sleep(0.1)
                _beep(1500, 150)
            elif state == DriverState.FATIGUE:
                _beep(600, 350)
