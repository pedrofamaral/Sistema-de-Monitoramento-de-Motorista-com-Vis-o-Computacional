"""Motor de decisão: transforma as saídas dos detectores numa lista de estados ativos."""
from __future__ import annotations

from enum import Enum


class DriverState(str, Enum):
    NORMAL = "NORMAL"
    DROWSY = "SONOLENCIA"
    PHONE = "CELULAR DETECTADO"


def decide(drowsy: bool, phone_in_use: bool) -> list[DriverState]:
    """Estados ativos; sonolência e celular podem ocorrer ao mesmo tempo."""
    states = []
    if drowsy:
        states.append(DriverState.DROWSY)
    if phone_in_use:
        states.append(DriverState.PHONE)
    return states or [DriverState.NORMAL]
