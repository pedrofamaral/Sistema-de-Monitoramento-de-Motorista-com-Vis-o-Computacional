"""HUD desenhado sobre o vídeo.

Layout inspirado no HUD do AlertEdge (AreebaAminn, licença MIT): faixa de status
no topo, FPS no canto e painel de métricas embaixo, adaptado aos nossos estados.
As fontes do OpenCV não têm acentos, por isso os textos da tela não usam acento.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..decision.engine import CRITICAL, DriverState
from ..detectors.face_landmarks import LEFT_EYE, MOUTH_CONTOUR, RIGHT_EYE
from ..pipeline import FrameResult

# Cores em BGR
GREEN = (60, 170, 60)
RED = (40, 40, 220)
AMBER = (0, 165, 255)
ORANGE = (0, 110, 240)
WHITE = (240, 240, 240)
GRAY = (150, 150, 150)
DARK = (22, 20, 18)
TITLE = (0, 215, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX
FULL_FRAME = (0.0, 0.0, 1.0, 1.0)


def _panel(frame, x1, y1, x2, y2, color, alpha):
    """Retângulo semitransparente."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return
    region = frame[y1:y2, x1:x2]
    frame[y1:y2, x1:x2] = cv2.addWeighted(np.full_like(region, color), alpha, region, 1 - alpha, 0)


def _rounded_rect(frame, pt1, pt2, color, thickness, radius):
    """Retangulo com cantos arredondados; thickness < 0 preenche."""
    x1, y1 = pt1
    x2, y2 = pt2
    if x2 <= x1 or y2 <= y1:
        return
    r = max(0, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))
    corners = ((x1 + r, y1 + r, 180, 270), (x2 - r, y1 + r, 270, 360),
               (x2 - r, y2 - r, 0, 90), (x1 + r, y2 - r, 90, 180))
    if thickness < 0:
        if r == 0:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
            return
        cv2.rectangle(frame, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(frame, (x1, y1 + r), (x2, y2 - r), color, -1)
        for cx, cy, a1, a2 in corners:
            cv2.ellipse(frame, (cx, cy), (r, r), 0, a1, a2, color, -1)
    else:
        if r == 0:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
            return
        cv2.line(frame, (x1 + r, y1), (x2 - r, y1), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x1 + r, y2), (x2 - r, y2), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x1, y1 + r), (x1, y2 - r), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x2, y1 + r), (x2, y2 - r), color, thickness, cv2.LINE_AA)
        for cx, cy, a1, a2 in corners:
            cv2.ellipse(frame, (cx, cy), (r, r), 0, a1, a2, color, thickness, cv2.LINE_AA)


def _corner_box(frame, x1, y1, x2, y2, color, thickness, s):
    """Caixa de deteccao estilo mira: so os 4 cantos, em L."""
    corner = max(int(10 * s), min((x2 - x1) // 3, (y2 - y1) // 3, int(22 * s)))
    for cx, cy, dx, dy in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(frame, (cx, cy), (cx + dx * corner, cy), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (cx, cy), (cx, cy + dy * corner), color, thickness, cv2.LINE_AA)


def _chip(frame, text, x, y, color, s, font_scale=0.5, text_color=WHITE):
    """Etiqueta arredondada e opaca, usada junto das caixas de deteccao."""
    thickness = max(1, round(1.5 * s))
    (tw, th), _ = cv2.getTextSize(text, FONT, font_scale * s, thickness)
    pad_x, pad_y = int(9 * s), int(6 * s)
    x2, y2 = x + tw + 2 * pad_x, y + th + 2 * pad_y
    _rounded_rect(frame, (x, y), (x2, y2), color, -1, (y2 - y) // 2)
    cv2.putText(frame, text, (x + pad_x, y2 - pad_y - 1), FONT, font_scale * s, text_color,
                thickness, cv2.LINE_AA)


def _pill(frame, text, x, y, color, s, font_scale=0.65, alpha=0.88,
          text_color=WHITE, align_right=False):
    """Faixa com texto; com align_right, x é a borda direita."""
    thickness = max(1, round(2 * s))
    (tw, th), _ = cv2.getTextSize(text, FONT, font_scale * s, thickness)
    pad, height = int(14 * s), int(38 * s)
    if align_right:
        x -= tw + 2 * pad
    _panel(frame, x, y, x + tw + 2 * pad, y + height, color, alpha)
    cv2.putText(frame, text, (x + pad, y + (height + th) // 2), FONT, font_scale * s,
                text_color, thickness, cv2.LINE_AA)


class Hud:
    def __init__(self, closed_seconds: float, phone_seconds: float, perclos_threshold: float,
                 min_observed_seconds: float, yawn_count: int,
                 roi: tuple[float, float, float, float] = FULL_FRAME):
        self.closed_seconds = closed_seconds
        self.phone_seconds = phone_seconds
        self.perclos_threshold = perclos_threshold
        self.min_observed_seconds = min_observed_seconds
        self.yawn_count = yawn_count
        self.roi = tuple(roi)

    @classmethod
    def from_config(cls, cfg: dict) -> "Hud":
        perclos = cfg["fatigue"]["perclos"]
        return cls(closed_seconds=cfg["drowsiness"]["closed_seconds"],
                   phone_seconds=cfg["phone"]["persistence"]["min_seconds"],
                   perclos_threshold=perclos["threshold"],
                   min_observed_seconds=perclos["min_observed_seconds"],
                   yawn_count=cfg["fatigue"]["yawn"]["count"],
                   roi=cfg["phone"]["filters"]["roi"])

    def draw(self, frame: np.ndarray, result: FrameResult, fps: float,
             show_mesh: bool = False) -> np.ndarray:
        """Desenha o resultado do pipeline sobre o frame."""
        h, w = frame.shape[:2]
        s = max(0.55, min(w / 1280.0, 1.0))  # escala relativa a 720p
        self._draw_overlays(frame, result.landmarks, result.eye, result.mouth, result.phone,
                            show_mesh, s)
        self._draw_status(frame, result.states, result.calibration, s)
        _pill(frame, f"{fps:.1f} FPS", w - int(16 * s), int(14 * s), DARK, s,
              font_scale=0.6, alpha=0.8, text_color=(200, 240, 200), align_right=True)
        self._draw_dock(frame, result.eye, result.fatigue, result.ear_threshold, result.calibrated,
                        result.phone, s)
        return frame

    def _draw_status(self, frame, states, calibration, s):
        if calibration is not None:
            text = f"CALIBRANDO: olhe para a camera com os olhos abertos ({calibration:.0%})"
            color = AMBER
        elif states == [DriverState.NORMAL]:
            text, color = "NORMAL", GREEN
        else:
            text = " | ".join(state.value for state in states)
            color = RED if CRITICAL.intersection(states) else ORANGE
        _pill(frame, text, int(16 * s), int(14 * s), color, s, font_scale=0.7)

    def _draw_overlays(self, frame, landmarks, eye, mouth, phone, show_mesh, s):
        h, w = frame.shape[:2]
        thin, thick = max(1, round(s)), max(2, round(2 * s))

        if self.roi != FULL_FRAME:
            x1, y1, x2, y2 = (int(self.roi[0] * w), int(self.roi[1] * h),
                              int(self.roi[2] * w), int(self.roi[3] * h))
            cv2.rectangle(frame, (x1, y1), (x2, y2), GRAY, thin)
            cv2.putText(frame, "regiao de interesse", (x1 + 6, y2 - 8), FONT, 0.5 * s,
                        GRAY, thin, cv2.LINE_AA)

        if landmarks is not None:
            if show_mesh:
                for x, y in landmarks.astype(int):
                    cv2.circle(frame, (x, y), 1, GRAY, -1)
            eye_color = RED if eye.eyes_closed else GREEN
            for indices in (RIGHT_EYE, LEFT_EYE):
                cv2.polylines(frame, [landmarks[list(indices)].astype(int)], True, eye_color,
                              thin, cv2.LINE_AA)
            cv2.polylines(frame, [landmarks[list(MOUTH_CONTOUR)].astype(int)], True,
                          ORANGE if mouth.yawning else GREEN, thin, cv2.LINE_AA)

        boxes = [(det, GRAY, thin, f"descartado {det.confidence:.2f}") for det in phone.rejected]
        box_color = RED if phone.in_use else AMBER
        boxes += [(det, box_color, thick, f"celular {det.confidence:.2f}") for det in phone.kept]
        for det, color, thickness, label in boxes:
            x1, y1, x2, y2 = det.box
            _corner_box(frame, x1, y1, x2, y2, color, thickness, s)
            chip_y = max(0, y1 - int(30 * s))
            text_color = WHITE if color is RED else DARK
            _chip(frame, label, x1, chip_y, color, s, text_color=text_color)

    def _draw_dock(self, frame, eye, fatigue, threshold, calibrated, phone, s):
        h, w = frame.shape[:2]
        top = h - int(92 * s)
        _panel(frame, 0, top, w, h, DARK, 0.85)
        cv2.line(frame, (0, top), (w, top), (65, 55, 50), max(1, round(1.5 * s)))
        col, pad = w // 4, int(18 * s)
        bar_w = col - 2 * pad

        # Coluna 1: abertura dos olhos
        if eye.ear is None:
            value, color = "SEM ROSTO", AMBER
        else:
            value, color = f"EAR {eye.ear:.2f}", RED if eye.eyes_closed else WHITE
        subtitle = f"limiar {threshold:.2f} ({'calibrado' if calibrated else 'fixo'})"
        self._column(frame, pad, top, "OLHOS", value, color, s, subtitle=subtitle)

        # Coluna 2: tempo de olhos fechados até a SONOLENCIA
        color = RED if eye.drowsy else (AMBER if eye.closed_for > 0 else WHITE)
        self._column(frame, col + pad, top, "OLHOS FECHADOS",
                     f"{eye.closed_for:.1f} s / {self.closed_seconds:.1f} s", color, s,
                     bar=(bar_w, eye.closed_for / self.closed_seconds))

        # Coluna 3: fadiga. Um critério por linha, com o limite escrito;
        # fica laranja só o critério que disparou.
        x = 2 * col + pad
        cv2.putText(frame, "FADIGA", (x, top + int(24 * s)), FONT, 0.5 * s, TITLE, 1, cv2.LINE_AA)
        if fatigue.ready:
            perclos_text = f"PERCLOS {fatigue.perclos:.0%} (limite {self.perclos_threshold:.0%})"
            perclos_color = ORANGE if fatigue.by_perclos else WHITE
        else:
            perclos_text = (f"PERCLOS (dados {fatigue.observed_seconds:.0f}/"
                            f"{self.min_observed_seconds:.0f} s)")
            perclos_color = GRAY
        yawn_text = f"Bocejos {fatigue.yawns} (limite {self.yawn_count})"
        yawn_color = ORANGE if fatigue.by_yawns else WHITE
        for text, color, y in ((perclos_text, perclos_color, 52), (yawn_text, yawn_color, 80)):
            cv2.putText(frame, text, (x, top + int(y * s)), FONT, 0.55 * s, color,
                        max(1, round(1.5 * s)), cv2.LINE_AA)

        # Coluna 4: celular
        color = RED if phone.in_use else (AMBER if phone.kept else WHITE)
        value = f"conf {phone.confidence:.2f}" if phone.kept else "nenhum"
        self._column(frame, 3 * col + pad, top, "CELULAR", value, color, s,
                     bar=(bar_w, phone.detected_for / self.phone_seconds))

    @staticmethod
    def _column(frame, x, top, title, value, color, s, subtitle=None, bar=None):
        cv2.putText(frame, title, (x, top + int(24 * s)), FONT, 0.5 * s, TITLE, 1, cv2.LINE_AA)
        cv2.putText(frame, value, (x, top + int(52 * s)), FONT, 0.68 * s, color,
                    max(1, round(2 * s)), cv2.LINE_AA)
        if subtitle:
            cv2.putText(frame, subtitle, (x, top + int(78 * s)), FONT, 0.46 * s, GRAY, 1,
                        cv2.LINE_AA)
        if bar:
            width, fraction = bar
            y1, y2 = top + int(64 * s), top + int(78 * s)
            radius = (y2 - y1) // 2
            _rounded_rect(frame, (x, y1), (x + width, y2), (52, 46, 42), -1, radius)
            fill = int(width * min(1.0, max(0.0, fraction)))
            if fill > 0:
                _rounded_rect(frame, (x, y1), (x + fill, y2), color, -1, radius)
                highlight_y2 = y1 + max(1, (y2 - y1) // 3)
                highlight = tuple(min(255, c + 45) for c in color)
                _rounded_rect(frame, (x, y1), (x + fill, highlight_y2), highlight, -1, radius)
            _rounded_rect(frame, (x, y1), (x + width, y2), (95, 86, 80), max(1, round(s)), radius)
