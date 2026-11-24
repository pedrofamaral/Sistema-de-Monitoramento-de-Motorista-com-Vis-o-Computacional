"""Regras do ramo do celular: filtros geométricos, região de interesse e persistência."""
from __future__ import annotations

from dataclasses import dataclass, field

from .phone_yolo import Detection
from .temporal import PersistenceFilter


@dataclass
class PhoneFilter:
    """Descarta detecções improváveis de serem um celular em uso pelo motorista."""

    min_box_size: float = 0.06  # menor lado da caixa, em fração da altura do frame
    # Faixa de altura/largura descartada: caixas quase quadradas costumam ser a mão.
    excluded_aspect: tuple[float, float] | None = (0.8, 1.2)
    # Região de interesse (x1, y1, x2, y2) em fração do frame; vale o centro da caixa.
    roi: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)

    def split(self, detections: list[Detection], frame_shape) -> tuple[list[Detection], list[Detection]]:
        """Separa as detecções em (aceitas, descartadas)."""
        height, width = frame_shape[:2]
        rx1, ry1, rx2, ry2 = (self.roi[0] * width, self.roi[1] * height,
                              self.roi[2] * width, self.roi[3] * height)
        kept, rejected = [], []
        for det in detections:
            x1, y1, x2, y2 = det.box
            box_w, box_h = max(1, x2 - x1), max(1, y2 - y1)
            aspect = box_h / box_w
            center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2

            big_enough = min(box_w, box_h) >= self.min_box_size * height
            square = (self.excluded_aspect is not None
                      and self.excluded_aspect[0] <= aspect <= self.excluded_aspect[1])
            in_roi = rx1 <= center_x <= rx2 and ry1 <= center_y <= ry2
            (kept if big_enough and not square and in_roi else rejected).append(det)
        return kept, rejected


@dataclass
class PhoneStatus:
    kept: list[Detection] = field(default_factory=list)      # detecções aceitas
    rejected: list[Detection] = field(default_factory=list)  # descartadas pelos filtros
    confidence: float = 0.0   # maior confiança entre as aceitas
    detected_for: float = 0.0 # segundos de detecção contínua
    in_use: bool = False      # CELULAR confirmado


class PhoneMonitor:
    """Celular aceito pelos filtros por pelo menos `min_seconds` = uso de celular."""

    def __init__(self, phone_filter: PhoneFilter, persistence: PersistenceFilter):
        self.filter = phone_filter
        self.persistence = persistence

    def update(self, detections: list[Detection], frame_shape, t: float) -> PhoneStatus:
        kept, rejected = self.filter.split(detections, frame_shape)
        in_use = self.persistence.update(bool(kept), t)
        return PhoneStatus(kept=kept, rejected=rejected,
                           confidence=max((d.confidence for d in kept), default=0.0),
                           detected_for=self.persistence.duration, in_use=in_use)
