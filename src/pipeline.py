"""Pipeline de um frame: percepção, regras temporais, decisão e eventos.

Usado pelo main.py (tempo real) e, na semana 4, pela avaliação nos vídeos de teste.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .capture import Frame
from .decision.engine import DriverState, decide
from .decision.events import Event, EventTracker
from .detectors.eyes import DrowsinessDetector, EARCalibrator, EyeStatus, average_ear
from .detectors.face_landmarks import FaceLandmarkDetector
from .detectors.fatigue import FatigueMonitor, FatigueStatus
from .detectors.mouth import MouthStatus, YawnDetector, mouth_aspect_ratio
from .detectors.phone import PhoneFilter, PhoneMonitor, PhoneStatus
from .detectors.phone_yolo import PhoneDetector
from .detectors.temporal import PersistenceFilter
from .utils.config import project_path


@dataclass
class FrameResult:
    timestamp: float
    landmarks: np.ndarray | None
    eye: EyeStatus
    mouth: MouthStatus
    fatigue: FatigueStatus
    phone: PhoneStatus
    states: list[DriverState]
    started: list[Event]        # eventos que começaram neste frame
    ended: list[Event]          # eventos que terminaram neste frame
    ear_threshold: float
    calibrated: bool
    calibration: float | None   # progresso da calibração (None = não está calibrando)
    timings: dict[str, float] = field(default_factory=dict)  # ms gastos em cada etapa


class DriverMonitor:
    def __init__(self, cfg: dict, calibrate: bool = True):
        eye_cfg, phone_cfg = cfg["drowsiness"], cfg["phone"]
        perclos_cfg, yawn_cfg = cfg["fatigue"]["perclos"], cfg["fatigue"]["yawn"]
        filters, persistence = phone_cfg["filters"], phone_cfg["persistence"]
        self._calib_cfg = eye_cfg["calibration"]
        self._yawn_window_min = yawn_cfg["window_seconds"] / 60

        self.face = FaceLandmarkDetector(project_path(cfg["models"]["face_landmarker"]), **cfg["face"])
        self.yolo = PhoneDetector(project_path(cfg["models"]["yolo"]), phone_cfg["class_name"],
                                  phone_cfg["confidence"], phone_cfg["imgsz"])
        # O YOLO é a etapa mais cara; roda a cada `interval_seconds` e, entre uma
        # rodada e outra, a última detecção vale. O intervalo precisa ser menor que a
        # tolerância a falhas da persistência, senão a contagem do celular zeraria
        # entre uma rodada e outra.
        self._yolo_interval = phone_cfg["interval_seconds"]
        if self._yolo_interval >= persistence["gap_tolerance"]:
            raise ValueError("phone.interval_seconds deve ser menor que phone.persistence.gap_tolerance")
        self._last_yolo_t: float | None = None
        self._detections = []
        self.drowsiness = DrowsinessDetector(eye_cfg["ear_threshold"], eye_cfg["hysteresis"],
                                             eye_cfg["closed_seconds"], eye_cfg["face_lost_grace"])
        self.yawn = YawnDetector(yawn_cfg["mar_threshold"], yawn_cfg["min_seconds"],
                                 yawn_cfg["gap_tolerance"])
        self.fatigue = FatigueMonitor(perclos_cfg["window_seconds"], perclos_cfg["threshold"],
                                      perclos_cfg["hysteresis"], perclos_cfg["min_observed_seconds"],
                                      yawn_cfg["window_seconds"], yawn_cfg["count"])
        aspect = filters["excluded_aspect"]
        self.phone = PhoneMonitor(
            PhoneFilter(min_box_size=filters["min_box_size"],
                        excluded_aspect=tuple(aspect) if aspect else None, roi=tuple(filters["roi"])),
            PersistenceFilter(persistence["min_seconds"], persistence["gap_tolerance"]))
        self.events = EventTracker()

        self.calibrator = self._new_calibrator() if calibrate and self._calib_cfg["enabled"] else None
        self.calibrated = False

    def _new_calibrator(self) -> EARCalibrator:
        return EARCalibrator(self._calib_cfg["seconds"], self._calib_cfg["ratio"])

    def recalibrate(self) -> None:
        self.calibrator, self.calibrated = self._new_calibrator(), False

    def process(self, frame: Frame) -> FrameResult:
        t = frame.timestamp
        clock = time.perf_counter()
        landmarks = self.face.detect(frame.image, t)
        timings = {"rosto": (time.perf_counter() - clock) * 1000}

        clock = time.perf_counter()
        if self._last_yolo_t is None or t - self._last_yolo_t >= self._yolo_interval:
            self._detections = self.yolo.detect(frame.image)
            self._last_yolo_t = t
        timings["celular"] = (time.perf_counter() - clock) * 1000

        clock = time.perf_counter()
        ear = average_ear(landmarks) if landmarks is not None else None

        # Olhos: calibração (se ativa) e sonolência
        if self.calibrator is not None:
            threshold = self.calibrator.update(ear, t)
            if threshold is not None:
                self.drowsiness.ear_threshold = threshold
                self.drowsiness.reset()
                self.calibrator, self.calibrated = None, True
                print(f"Calibração concluída: limiar do EAR = {threshold:.3f}")
        calibration = self.calibrator.progress if self.calibrator is not None else None
        eye = (self.drowsiness.update(ear, t) if calibration is None  # sonolência desligada calibrando
               else EyeStatus(ear=ear, eyes_closed=False, closed_for=0.0, drowsy=False))

        # Boca e fadiga (o PERCLOS só usa frames com rosto e já calibrados)
        mouth = self.yawn.update(mouth_aspect_ratio(landmarks) if landmarks is not None else None, t)
        eyes_known = ear is not None and calibration is None
        fatigue = self.fatigue.update(eye.eyes_closed if eyes_known else None, mouth.yawn_started, t)

        # Celular (filtros e persistência sobre a última detecção do YOLO)
        phone = self.phone.update(self._detections, frame.image.shape, t)

        # Decisão e eventos
        states = decide(drowsy=eye.drowsy, phone_in_use=phone.in_use, fatigued=fatigue.fatigued)
        values = {DriverState.DROWSY: ear,
                  DriverState.PHONE: phone.confidence if phone.kept else None,
                  DriverState.FATIGUE: fatigue.perclos if fatigue.ready else None}
        details = {DriverState.DROWSY: f"Olhos fechados há {eye.closed_for:.1f} s",
                   DriverState.PHONE: f"Celular com confiança {phone.confidence:.2f}",
                   DriverState.FATIGUE: self._fatigue_reason(fatigue)}
        started, ended = self.events.update(states, t, values, details)
        timings["regras"] = (time.perf_counter() - clock) * 1000

        return FrameResult(timestamp=t, landmarks=landmarks, eye=eye, mouth=mouth, fatigue=fatigue,
                           phone=phone, states=states, started=started, ended=ended,
                           ear_threshold=self.drowsiness.ear_threshold, calibrated=self.calibrated,
                           calibration=calibration, timings=timings)

    def _fatigue_reason(self, fatigue: FatigueStatus) -> str:
        reasons = []
        if fatigue.by_perclos:
            reasons.append(f"olhos fechados {fatigue.perclos:.0%} do tempo no último minuto (PERCLOS)")
        if fatigue.by_yawns:
            reasons.append(f"{fatigue.yawns} bocejos nos últimos {self._yawn_window_min:.0f} min")
        text = " e ".join(reasons)
        return text[:1].upper() + text[1:]

    def finish(self, t: float) -> list[Event]:
        """Encerra os eventos abertos (fim do vídeo ou programa fechado)."""
        return self.events.close(t)

    def close(self) -> None:
        self.face.close()
