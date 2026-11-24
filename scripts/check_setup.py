"""Verificação da semana 1: webcam, landmarks faciais e YOLO funcionando juntos.

Uso:
  python scripts/check_setup.py                       # webcam do settings.yaml
  python scripts/check_setup.py --source video.mp4    # arquivo de vídeo
  python scripts/check_setup.py --frames 150 --no-window   # teste rápido sem janela

Teclas: q ou ESC sai; m mostra/esconde a malha completa de landmarks.
"""
import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402

from src.capture import VideoSource  # noqa: E402
from src.detectors.face_landmarks import LEFT_EYE, RIGHT_EYE, FaceLandmarkDetector  # noqa: E402
from src.detectors.phone_yolo import PhoneDetector  # noqa: E402
from src.utils.config import load_config, project_path  # noqa: E402
from src.utils.fps import FPSCounter  # noqa: E402

WINDOW = "Verificacao - semana 1"
GREEN, RED, GRAY, WHITE = (80, 220, 80), (60, 60, 230), (160, 160, 160), (255, 255, 255)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--source", help="índice da webcam ou caminho do vídeo (padrão: settings.yaml)")
    p.add_argument("--frames", type=int, default=0, help="para depois de N frames (0 = sem limite)")
    p.add_argument("--no-window", action="store_true", help="não abre janela (teste automático)")
    p.add_argument("--skip-face", action="store_true", help="desliga o MediaPipe")
    p.add_argument("--skip-yolo", action="store_true", help="desliga o YOLO")
    return p.parse_args()


def draw(img, landmarks, detections, fps, show_mesh):
    if landmarks is not None:
        if show_mesh:
            for x, y in landmarks.astype(int):
                cv2.circle(img, (x, y), 1, GRAY, -1)
        for eye in (RIGHT_EYE, LEFT_EYE):
            pts = landmarks[list(eye)].astype(int)
            cv2.polylines(img, [pts], isClosed=True, color=GREEN, thickness=1)
            for x, y in pts:
                cv2.circle(img, (x, y), 2, GREEN, -1)

    for det in detections:
        x1, y1, x2, y2 = det.box
        cv2.rectangle(img, (x1, y1), (x2, y2), RED, 2)
        cv2.putText(img, f"celular {det.confidence:.2f}", (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 2, cv2.LINE_AA)

    lines = [f"FPS: {fps:.1f}",
             f"Rosto: {'SIM' if landmarks is not None else 'NAO'}",
             f"Celulares: {len(detections)}"]
    for i, text in enumerate(lines):
        cv2.putText(img, text, (12, 28 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2, cv2.LINE_AA)


def summarize(name, values):
    if not values:
        return f"  {name}: desligado"
    return (f"  {name}: mediana {statistics.median(values):.1f} ms, "
            f"máx {max(values):.1f} ms (o 1º frame inclui o aquecimento do modelo)")


def main():
    args = parse_args()
    cfg = load_config()
    cap_cfg = cfg["capture"]
    source = args.source if args.source is not None else cap_cfg["source"]

    face = None if args.skip_face else FaceLandmarkDetector(
        project_path(cfg["models"]["face_landmarker"]), **cfg["face"])
    phone = None if args.skip_yolo else PhoneDetector(
        project_path(cfg["models"]["yolo"]), cfg["phone"]["class_name"],
        cfg["phone"]["confidence"], cfg["phone"]["imgsz"])

    fps = FPSCounter()
    face_ms, yolo_ms = [], []
    n_frames = n_face = n_phone = 0
    show_mesh = False
    t_first = t_last = None

    with VideoSource(source, cap_cfg["width"], cap_cfg["height"], cap_cfg["mirror"]) as video:
        width, height = video.resolution
        print(f"Fonte: {source} ({width}x{height})")
        for frame in video:
            landmarks, detections = None, []
            if face:
                t0 = time.perf_counter()
                landmarks = face.detect(frame.image, frame.timestamp)
                face_ms.append((time.perf_counter() - t0) * 1000)
            if phone:
                t0 = time.perf_counter()
                detections = phone.detect(frame.image)
                yolo_ms.append((time.perf_counter() - t0) * 1000)

            n_frames += 1
            n_face += landmarks is not None
            n_phone += bool(detections)
            current_fps = fps.tick()
            t_last = time.perf_counter()
            t_first = t_first or t_last

            if not args.no_window:
                draw(frame.image, landmarks, detections, current_fps, show_mesh)
                cv2.imshow(WINDOW, frame.image)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("m"):
                    show_mesh = not show_mesh
            if args.frames and n_frames >= args.frames:
                break

    if face:
        face.close()
    cv2.destroyAllWindows()

    if n_frames == 0:
        print("Nenhum frame lido. Confira a fonte de vídeo (índice da webcam ou caminho).")
        return
    avg_fps = (n_frames - 1) / (t_last - t_first) if n_frames > 1 and t_last > t_first else 0.0
    print(f"\nFrames processados: {n_frames} | FPS médio: {avg_fps:.1f}")
    print(f"Rosto encontrado em {100 * n_face / n_frames:.0f}% dos frames")
    print(f"Celular detectado em {n_phone} frames")
    print("Latência por frame:")
    print(summarize("MediaPipe", face_ms))
    print(summarize("YOLO", yolo_ms))


if __name__ == "__main__":
    main()
