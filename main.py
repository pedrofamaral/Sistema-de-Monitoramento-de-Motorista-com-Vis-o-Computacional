"""Monitor de motorista: sonolência, fadiga e uso de celular em tempo real.

Uso:
  python main.py                       # webcam do settings.yaml
  python main.py --source video.mp4    # arquivo de vídeo
  python main.py --no-calibration      # usa o limiar fixo do EAR do settings.yaml
  python main.py --no-alerts           # sem som e sem Telegram (útil nos vídeos de teste)

Teclas: q ou ESC sai · c recalibra os olhos · m mostra/esconde a malha do rosto
Os eventos são gravados em logs/events.csv.
"""
import argparse

import cv2

from src.capture import VideoSource
from src.interface.alerts import AlertSound
from src.interface.event_log import EventLogger
from src.interface.hud import Hud
from src.interface.telegram import TelegramNotifier
from src.pipeline import DriverMonitor
from src.utils.config import apply_overrides, load_config, load_secrets, project_path
from src.utils.fps import FPSCounter

WINDOW = "Monitor de Motorista"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--source", help="índice da webcam ou caminho do vídeo (padrão: settings.yaml)")
    p.add_argument("--no-calibration", action="store_true",
                   help="não calibra; usa drowsiness.ear_threshold do settings.yaml")
    p.add_argument("--no-alerts", action="store_true", help="desliga o som e o Telegram")
    p.add_argument("--no-window", action="store_true", help="processa sem abrir janela")
    p.add_argument("--frames", type=int, default=0, help="para depois de N frames (0 = sem limite)")
    p.add_argument("--set", action="append", default=[], metavar="CHAVE=VALOR",
                   help="troca um parâmetro do settings.yaml, ex.: --set drowsiness.closed_seconds=3")
    return p.parse_args()


def build_notifier(cfg: dict) -> TelegramNotifier | None:
    tg_cfg = cfg["telegram"]
    if not tg_cfg["enabled"]:
        return None
    secrets = load_secrets().get("telegram", {})
    if not secrets.get("token") or not secrets.get("chat_id"):
        print("[telegram] desligado: rode 'python scripts/telegram_setup.py' para configurar o bot")
        return None
    print("[telegram] ligado: os alertas serão enviados ao chat configurado")
    return TelegramNotifier(secrets["token"], secrets["chat_id"], tg_cfg["states"],
                            tg_cfg["cooldown_seconds"], tg_cfg["send_photo"])


def main():
    args = parse_args()
    cfg = apply_overrides(load_config(), args.set)
    cap_cfg = cfg["capture"]
    source = args.source if args.source is not None else cap_cfg["source"]

    monitor = DriverMonitor(cfg, calibrate=not args.no_calibration)
    hud = Hud.from_config(cfg)
    log = EventLogger(project_path(cfg["events"]["csv"]), source=str(source))
    sound = AlertSound() if cfg["alerts"]["sound"] and not args.no_alerts else None
    notifier = None if args.no_alerts else build_notifier(cfg)
    fps = FPSCounter()
    show_mesh = False
    last_t = 0.0

    if not args.no_window:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

    try:
        with VideoSource(source, cap_cfg["width"], cap_cfg["height"], cap_cfg["mirror"],
                         cap_cfg["max_fps"]) as video:
            for frame in video:
                result = monitor.process(frame)
                last_t = result.timestamp
                for event in result.ended:
                    log.write(event)
                if sound:
                    sound.update(result.states, result.started)

                current_fps = fps.tick()
                if not args.no_window:
                    hud.draw(frame.image, result, current_fps, show_mesh)
                if notifier:  # depois do HUD, para a foto mostrar o alerta
                    for event in result.started:
                        notifier.notify(event, frame.image)

                if not args.no_window:
                    cv2.imshow(WINDOW, frame.image)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                        break
                    if key == ord("c"):
                        monitor.recalibrate()
                    if key == ord("m"):
                        show_mesh = not show_mesh
                if args.frames and frame.index + 1 >= args.frames:
                    break
    finally:
        # Eventos ainda abertos (ex.: fechou o programa no meio de um alerta) também são salvos.
        for event in monitor.finish(last_t):
            log.write(event)
        log.close()
        if sound:
            sound.close()
        if notifier:
            notifier.close()
        monitor.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
