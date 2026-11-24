"""Motor de decisão: transforma as saídas dos detectores numa lista de estados ativos."""
from __future__ import annotations

from enum import Enum


class DriverState(str, Enum):
    NORMAL = "NORMAL"
    DROWSY = "SONOLENCIA"          # perigo imediato: olhos fechados agora
    PHONE = "CELULAR DETECTADO"    # perigo imediato: distração
    FATIGUE = "FADIGA"             # aviso: cansaço acumulado nos últimos minutos


CRITICAL = {DriverState.DROWSY, DriverState.PHONE}


def decide(*, drowsy: bool, phone_in_use: bool, fatigued: bool) -> list[DriverState]:
    """Estados ativos, do mais grave ao menos grave; podem ocorrer juntos."""
    states = [state for state, active in ((DriverState.DROWSY, drowsy),
                                          (DriverState.PHONE, phone_in_use),
                                          (DriverState.FATIGUE, fatigued)) if active]
    return states or [DriverState.NORMAL]
