"""Notificação remota pelo Telegram: avisa a "central" quando um alerta começa.

O envio acontece numa thread separada, então a internet lenta não trava o vídeo.
O token e o chat_id ficam em config/secrets.yaml, que não vai para o git.
"""
from __future__ import annotations

import queue
import threading

import cv2
import numpy as np
import requests

from ..decision.engine import DriverState
from ..decision.events import Event

API_URL = "https://api.telegram.org/bot{token}/{method}"

TITLES = {
    DriverState.DROWSY: "🚨 SONOLÊNCIA: motorista pode ter dormido",
    DriverState.PHONE: "📱 USO DE CELULAR",
    DriverState.FATIGUE: "⚠️ FADIGA: sinais de cansaço",
}


def telegram_api(token: str, method: str, timeout: float = 15, **kwargs) -> dict:
    """Chama um método da API de bots e retorna o campo `result` (ou levanta erro).

    Os erros nunca trazem a URL, porque ela contém o token.
    """
    try:
        response = requests.post(API_URL.format(token=token, method=method), timeout=timeout, **kwargs)
        data = response.json()
    except requests.RequestException as error:
        raise RuntimeError(f"sem conexão com o Telegram ({type(error).__name__})") from None
    except ValueError:
        raise RuntimeError(f"resposta inválida do Telegram (HTTP {response.status_code})") from None
    if not data.get("ok"):
        raise RuntimeError(f"Telegram recusou {method}: {data.get('description', response.status_code)}")
    return data["result"]


def format_message(event: Event) -> str:
    return (f"{TITLES.get(event.state, event.state.value)}\n{event.detail}\n"
            f"{event.started_at:%d/%m/%Y %H:%M:%S}")


class TelegramNotifier:
    def __init__(self, token: str, chat_id: str | int, states: list[str],
                 cooldown_seconds: float, send_photo: bool = True):
        self.token = token
        self.chat_id = chat_id
        self.states = {DriverState(value) for value in states}
        self.cooldown_seconds = cooldown_seconds
        self.send_photo = send_photo
        self._last_sent: dict[DriverState, float] = {}
        self._queue: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def notify(self, event: Event, image_bgr: np.ndarray | None = None) -> bool:
        """Agenda o aviso de um evento que começou; retorna False se foi filtrado."""
        if event.state not in self.states:
            return False
        last = self._last_sent.get(event.state)
        if last is not None and event.start - last < self.cooldown_seconds:
            return False  # evita uma rajada de mensagens do mesmo tipo
        self._last_sent[event.state] = event.start

        photo = None
        if self.send_photo and image_bgr is not None:
            ok, jpg = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
            photo = jpg.tobytes() if ok else None
        self._queue.put((format_message(event), photo))
        return True

    def close(self, timeout: float = 10) -> None:
        """Espera os avisos pendentes serem enviados (até `timeout` segundos)."""
        self._queue.put(None)
        self._thread.join(timeout)

    def _run(self) -> None:
        while (item := self._queue.get()) is not None:
            text, photo = item
            try:
                if photo is not None:
                    telegram_api(self.token, "sendPhoto", data={"chat_id": self.chat_id, "caption": text},
                                 files={"photo": ("alerta.jpg", photo, "image/jpeg")})
                else:
                    telegram_api(self.token, "sendMessage", json={"chat_id": self.chat_id, "text": text})
            except Exception as error:  # sem internet, token errado etc.: avisa e segue
                print(f"[telegram] falha ao enviar: {error}")
