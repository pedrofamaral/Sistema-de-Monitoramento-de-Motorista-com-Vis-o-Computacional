"""Monitor de motorista: detecção de sonolência e uso de celular em tempo real.

Uso:
  python main.py                       # webcam do settings.yaml
  python main.py --source video.mp4    # arquivo de vídeo
  python main.py --no-calibration      # usa o limiar fixo do EAR do settings.yaml

Teclas: q ou ESC sai · c recalibra os olhos · m mostra/esconde a malha do rosto
"""
import argparse

import cv2

from src.capture import VideoSource
from src.decision.engine import decide
from src.detectors.eyes import DrowsinessDetector, EARCalibrator, EyeStatus, average_ear
from src.detectors.face_landmarks import FaceLandmarkDetector
from src.detectors.fatigue import FatigueMonitor
from src.detectors.mouth import YawnDetector, mouth_aspect_ratio
from src.detectors.phone import PhoneFilter, PhoneMonitor
from src.detectors.phone_yolo import PhoneDetector
from src.detectors.temporal import PersistenceFilter
from src.interface.hud import Hud
from src.utils.config import load_config, project_path
from src.utils.fps import FPSCounter

WINDOW = "Monitor de Motorista"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--source", help="índice da webcam ou caminho do vídeo (padrão: settings.yaml)")
    p.add_argument("--no-calibration", action="store_true",
                   help="não calibra; usa drowsiness.ear_threshold do settings.yaml")
    p.add_argument("--no-window", action="store_true", help="processa sem abrir janela")
    p.add_argument("--frames", type=int, default=0, help="para depois de N frames (0 = sem limite)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config()
    cap_cfg, eye_cfg, phone_cfg = cfg["capture"], cfg["drowsiness"], cfg["phone"]
    calib_cfg = eye_cfg["calibration"]
    perclos_cfg, yawn_cfg = cfg["fatigue"]["perclos"], cfg["fatigue"]["yawn"]

    face = FaceLandmarkDetector(project_path(cfg["models"]["face_landmarker"]), **cfg["face"])
    yolo = PhoneDetector(project_path(cfg["models"]["yolo"]), phone_cfg["class_name"],
                         phone_cfg["confidence"], phone_cfg["imgsz"])
    drowsiness = DrowsinessDetector(eye_cfg["ear_threshold"], eye_cfg["hysteresis"],
                                    eye_cfg["closed_seconds"], eye_cfg["face_lost_grace"])
    filters = phone_cfg["filters"]
    phone = PhoneMonitor(
        PhoneFilter(min_box_size=filters["min_box_size"],
                    excluded_aspect=tuple(filters["excluded_aspect"]) if filters["excluded_aspect"] else None,
                    roi=tuple(filters["roi"])),
        PersistenceFilter(phone_cfg["persistence"]["min_seconds"],
                          phone_cfg["persistence"]["gap_tolerance"]))
    yawn = YawnDetector(yawn_cfg["mar_threshold"], yawn_cfg["min_seconds"], yawn_cfg["gap_tolerance"])
    fatigue = FatigueMonitor(perclos_cfg["window_seconds"], perclos_cfg["threshold"],
                             perclos_cfg["hysteresis"], perclos_cfg["min_observed_seconds"],
                             yawn_cfg["window_seconds"], yawn_cfg["count"])
    hud = Hud(eye_cfg["closed_seconds"], phone_cfg["persistence"]["min_seconds"],
              perclos_cfg["threshold"], perclos_cfg["min_observed_seconds"], yawn_cfg["count"],
              filters["roi"])
    fps = FPSCounter()

    def new_calibrator():
        return EARCalibrator(calib_cfg["seconds"], calib_cfg["ratio"])

    calibrator = new_calibrator() if calib_cfg["enabled"] and not args.no_calibration else None
    calibrated = False
    show_mesh = False

    if not args.no_window:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

    source = args.source if args.source is not None else cap_cfg["source"]
    with VideoSource(source, cap_cfg["width"], cap_cfg["height"], cap_cfg["mirror"]) as video:
        for frame in video:
            t = frame.timestamp
            landmarks = face.detect(frame.image, t)
            ear = average_ear(landmarks) if landmarks is not None else None

            if calibrator is not None:
                threshold = calibrator.update(ear, t)
                if threshold is not None:
                    drowsiness.ear_threshold = threshold
                    drowsiness.reset()
                    calibrator, calibrated = None, True
                    print(f"Calibração concluída: limiar do EAR = {threshold:.3f}")
            # Durante a calibração a sonolência fica desligada.
            eye = (drowsiness.update(ear, t) if calibrator is None
                   else EyeStatus(ear=ear, eyes_closed=False, closed_for=0.0, drowsy=False))

            mouth = yawn.update(mouth_aspect_ratio(landmarks) if landmarks is not None else None, t)
            # O PERCLOS só usa frames com rosto e já calibrados.
            eyes_known = ear is not None and calibrator is None
            fatigue_status = fatigue.update(eye.eyes_closed if eyes_known else None,
                                            mouth.yawn_started, t)

            phone_status = phone.update(yolo.detect(frame.image), frame.image.shape, t)
            states = decide(drowsy=eye.drowsy, phone_in_use=phone_status.in_use,
                            fatigued=fatigue_status.fatigued)
            current_fps = fps.tick()

            if not args.no_window:
                hud.draw(frame.image, states=states, fps=current_fps, landmarks=landmarks,
                         eye=eye, mouth=mouth, fatigue=fatigue_status,
                         threshold=drowsiness.ear_threshold, calibrated=calibrated,
                         calibration=calibrator.progress if calibrator else None,
                         phone=phone_status, show_mesh=show_mesh)
                cv2.imshow(WINDOW, frame.image)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key == ord("c"):
                    calibrator, calibrated = new_calibrator(), False
                if key == ord("m"):
                    show_mesh = not show_mesh
            if args.frames and frame.index + 1 >= args.frames:
                break

    face.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
