"""Registro de eventos em CSV (events.csv), uma linha por evento concluído."""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

from ..decision.events import Event

FIELDS = ["sessao", "fonte", "evento", "inicio_s", "fim_s", "duracao_s", "valor",
          "detalhe", "data_hora_inicio"]


class EventLogger:
    """Acrescenta eventos ao CSV; cria o arquivo e o cabeçalho se não existirem.

    inicio_s e fim_s são os timestamps dos frames (posição no vídeo, ou tempo desde
    a abertura da webcam), para comparar com os vídeos rotulados na avaliação.
    valor: EAR mínimo (SONOLENCIA), confiança máxima (CELULAR), PERCLOS máximo (FADIGA).
    """

    def __init__(self, path: str | Path, source: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not self.path.exists() or self.path.stat().st_size == 0
        # utf-8-sig: o Excel abre os acentos corretamente
        self._file = open(self.path, "a", newline="", encoding="utf-8-sig" if new_file else "utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=FIELDS)
        if new_file:
            self._writer.writeheader()
        self.session = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.source = source

    def write(self, event: Event) -> None:
        self._writer.writerow({
            "sessao": self.session,
            "fonte": self.source,
            "evento": event.state.value,
            "inicio_s": f"{event.start:.2f}",
            "fim_s": f"{event.end:.2f}" if event.end is not None else "",
            "duracao_s": f"{event.duration:.2f}",
            "valor": f"{event.peak:.3f}" if event.peak is not None else "",
            "detalhe": event.detail,
            "data_hora_inicio": event.started_at.strftime("%Y-%m-%d %H:%M:%S"),
        })
        self._file.flush()  # o evento fica salvo mesmo se o programa fechar de repente

    def close(self) -> None:
        self._file.close()
