"""Captura de frames de webcam ou arquivo de vídeo, com o timestamp de cada frame."""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Frame:
    image: np.ndarray  # imagem BGR
    timestamp: float   # segundos desde o início da captura
    index: int         # número do frame (0, 1, 2, ...)


class VideoSource:
    """Abre uma webcam (índice inteiro) ou um arquivo de vídeo (caminho).

    O timestamp vem do relógio do sistema, na webcam, e da posição dentro do
    vídeo, em arquivos. Assim as regras temporais (ex.: olhos fechados por 2 s)
    valem igual nos dois casos, e o resultado de um vídeo de teste não depende
    da velocidade do computador que o processa.
    """

    def __init__(self, source: int | str, width: int | None = None,
                 height: int | None = None, mirror: bool = False, max_fps: float = 0):
        """`max_fps` > 0 descarta frames para processar no máximo essa taxa.

        Os frames descartados não são decodificados (grab sem retrieve), então
        quase não custam CPU. Vale para webcam e para arquivo (pelo timestamp),
        assim um vídeo de teste é processado como seria ao vivo.
        """
        if isinstance(source, str) and source.isdigit():
            source = int(source)
        self.is_webcam = isinstance(source, int)
        self.mirror = mirror
        self._min_interval = 1.0 / max_fps if max_fps else 0.0
        self._last_ts: float | None = None
        self.skipped = 0  # frames descartados pelo limite de FPS

        if self.is_webcam:
            # No Windows o backend DirectShow abre a webcam bem mais rápido que o padrão.
            backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
            self._cap = cv2.VideoCapture(source, backend)
            if width and height:
                # MJPG permite 720p com FPS alto na maioria das webcams USB.
                self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        else:
            if not Path(source).exists():
                raise FileNotFoundError(f"Vídeo não encontrado: {source}")
            self._cap = cv2.VideoCapture(str(source))

        if not self._cap.isOpened():
            raise RuntimeError(f"Não foi possível abrir a fonte de vídeo: {source}")

        self._fps_file = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._start = time.perf_counter()
        self._index = 0

    @property
    def resolution(self) -> tuple[int, int]:
        return (int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    def read(self) -> Frame | None:
        """Lê o próximo frame; retorna None no fim do vídeo ou em erro da câmera."""
        result = self._read_webcam() if self.is_webcam else self._read_file()
        if result is None:
            return None
        image, timestamp = result
        self._last_ts = timestamp
        if self.mirror:
            image = cv2.flip(image, 1)
        frame = Frame(image=image, timestamp=timestamp, index=self._index)
        self._index += 1
        return frame

    def _read_webcam(self):
        # Limite de FPS: dorme o tempo que falta. Não dá para ler e descartar em laço,
        # porque o grab() da webcam não espera o próximo frame e o laço gastaria CPU à toa.
        if self._min_interval and self._last_ts is not None:
            wait = self._min_interval - (time.perf_counter() - self._start - self._last_ts)
            if wait > 0:
                time.sleep(wait)
        ok, image = self._cap.read()
        return (image, time.perf_counter() - self._start) if ok else None

    def _read_file(self):
        # Limite de FPS: descarta pelo timestamp do vídeo, sem decodificar (grab sem retrieve).
        while self._cap.grab():
            timestamp = self._file_timestamp()
            # Margem de 10% para pequenas variações no intervalo entre frames.
            if self._last_ts is not None and timestamp - self._last_ts < 0.9 * self._min_interval:
                self._index += 1
                self.skipped += 1
                continue
            ok, image = self._cap.retrieve()
            return (image, timestamp) if ok else None
        return None

    def _file_timestamp(self) -> float:
        # Posição real no vídeo (funciona com taxa de quadros variável, comum em
        # gravações de celular); usa índice/FPS como reserva.
        pos_ms = self._cap.get(cv2.CAP_PROP_POS_MSEC)
        return pos_ms / 1000.0 if pos_ms > 0 or self._index == 0 else self._index / self._fps_file

    def release(self) -> None:
        self._cap.release()

    def __enter__(self) -> VideoSource:
        return self

    def __exit__(self, *exc) -> None:
        self.release()

    def __iter__(self):
        while (frame := self.read()) is not None:
            yield frame
