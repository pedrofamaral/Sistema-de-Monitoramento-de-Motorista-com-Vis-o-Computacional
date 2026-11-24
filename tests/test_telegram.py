from datetime import datetime

import numpy as np
import pytest
import requests

from src.decision.engine import DriverState
from src.decision.events import Event
from src.interface import telegram
from src.interface.telegram import TelegramNotifier, format_message, telegram_api

ALL_STATES = ["SONOLENCIA", "CELULAR DETECTADO", "FADIGA"]


def event(state, start):
    return Event(state, start, datetime(2026, 11, 24, 15, 42, 10), detail="detalhe")


@pytest.fixture
def sent(monkeypatch):
    """Substitui a API do Telegram e guarda as chamadas."""
    calls = []
    monkeypatch.setattr(telegram, "telegram_api", lambda token, method, **kw: calls.append((method, kw)))
    return calls


def test_sends_photo_with_caption(sent):
    notifier = TelegramNotifier("t", 42, ALL_STATES, cooldown_seconds=60)
    assert notifier.notify(event(DriverState.DROWSY, 5.0), np.zeros((48, 64, 3), np.uint8))
    notifier.close()
    method, kwargs = sent[0]
    assert method == "sendPhoto"
    assert kwargs["data"]["chat_id"] == 42 and "SONOLÊNCIA" in kwargs["data"]["caption"]


def test_text_only_when_photo_disabled(sent):
    notifier = TelegramNotifier("t", 42, ALL_STATES, cooldown_seconds=60, send_photo=False)
    notifier.notify(event(DriverState.FATIGUE, 5.0), np.zeros((48, 64, 3), np.uint8))
    notifier.close()
    assert sent[0][0] == "sendMessage"


def test_filters_states_and_applies_cooldown(sent):
    notifier = TelegramNotifier("t", 42, ["SONOLENCIA"], cooldown_seconds=60)
    assert not notifier.notify(event(DriverState.PHONE, 0.0))        # estado fora da lista
    assert notifier.notify(event(DriverState.DROWSY, 10.0))
    assert not notifier.notify(event(DriverState.DROWSY, 40.0))      # dentro de 60 s
    assert notifier.notify(event(DriverState.DROWSY, 75.0))
    notifier.close()
    assert len(sent) == 2


def test_send_failure_does_not_crash(monkeypatch, capsys):
    def fail(*args, **kwargs):
        raise RuntimeError("sem conexão com o Telegram")
    monkeypatch.setattr(telegram, "telegram_api", fail)
    notifier = TelegramNotifier("t", 42, ALL_STATES, cooldown_seconds=0)
    notifier.notify(event(DriverState.DROWSY, 0.0))
    notifier.close()
    assert "falha ao enviar" in capsys.readouterr().out


def test_errors_never_reveal_the_token(monkeypatch):
    def offline(url, **kwargs):
        raise requests.ConnectionError(f"Max retries exceeded with url: {url}")
    monkeypatch.setattr(telegram.requests, "post", offline)
    with pytest.raises(RuntimeError) as error:
        telegram_api("123456:SEGREDO", "sendMessage")
    assert "SEGREDO" not in str(error.value)


def test_message_text():
    text = format_message(event(DriverState.PHONE, 0.0))
    assert text.splitlines() == ["📱 USO DE CELULAR", "detalhe", "24/11/2026 15:42:10"]
