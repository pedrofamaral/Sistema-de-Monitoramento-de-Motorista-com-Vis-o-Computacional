"""Mede o custo computacional do pipeline completo: uso de CPU, FPS e tempo por etapa.

Em arquivos de vídeo, processa no ritmo real (como uma webcam): espera o horário
de cada frame e descarta frames quando atrasa. Assim o uso de CPU medido é o que
o programa gastaria rodando ao vivo. Inclui o desenho do HUD (sem abrir janela).

Uso:
  python scripts/benchmark.py --source video.mp4
  python scripts/benchmark.py --source video.mp4 --set phone.imgsz=320 --label "imgsz 320"
  python scripts/benchmark.py --source 0 --seconds 30     # webcam
  python scripts/benchmark.py --source video.mp4 --csv logs/benchmark.csv   # guarda o resultado
"""
import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import psutil  # noqa: E402

from src.capture import VideoSource  # noqa: E402
from src.interface.hud import Hud  # noqa: E402
from src.pipeline import DriverMonitor  # noqa: E402
from src.utils.config import apply_overrides, load_config  # noqa: E402
from src.utils.fps import FPSCounter  # noqa: E402

WARMUP_FRAMES = 15   # descartados das estatísticas (carregamento e aquecimento dos modelos)
MAX_LATENESS = 0.05  # s de atraso a partir do qual o frame é descartado (como numa webcam)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--source", help="vídeo ou índice da webcam (padrão: settings.yaml)")
    p.add_argument("--seconds", type=float, default=0, help="duração máxima da medição")
    p.add_argument("--set", action="append", default=[], metavar="CHAVE=VALOR",
                   help="troca um parâmetro do settings.yaml (pode repetir)")
    p.add_argument("--label", default="", help="nome da configuração no resultado")
    p.add_argument("--csv", help="acrescenta o resultado a este CSV")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = apply_overrides(load_config(), args.set)
    cap = cfg["capture"]
    source = args.source if args.source is not None else cap["source"]

    monitor = DriverMonitor(cfg)
    hud, fps = Hud.from_config(cfg), FPSCounter()
    proc = psutil.Process()
    stages: dict[str, list[float]] = {"rosto": [], "celular": [], "regras": []}
    totals, yolo_runs = [], []
    processed = dropped = 0
    cpu_start = wall_start = clock_origin = None

    with VideoSource(source, cap["width"], cap["height"], cap["mirror"], cap["max_fps"]) as video:
        width, height = video.resolution
        for frame in video:
            if clock_origin is None:
                clock_origin = time.perf_counter() - frame.timestamp
            if not video.is_webcam:  # ritmo real
                lateness = time.perf_counter() - (clock_origin + frame.timestamp)
                if lateness < 0:
                    time.sleep(-lateness)
                elif lateness > MAX_LATENESS and processed > WARMUP_FRAMES:
                    dropped += 1
                    continue

            start = time.perf_counter()
            result = monitor.process(frame)
            hud.draw(frame.image, result, fps.tick())
            elapsed = (time.perf_counter() - start) * 1000
            processed += 1

            if processed == WARMUP_FRAMES:
                cpu_start, wall_start = proc.cpu_times(), time.perf_counter()
            elif processed > WARMUP_FRAMES:
                totals.append(elapsed)
                for name, value in result.timings.items():
                    stages[name].append(value)
                if result.timings["celular"] > 0.5:  # o YOLO rodou neste frame
                    yolo_runs.append(result.timings["celular"])
                if args.seconds and time.perf_counter() - wall_start >= args.seconds:
                    break
    monitor.close()

    if not totals:
        sys.exit("Poucos frames para medir.")
    wall = time.perf_counter() - wall_start
    cpu_end = proc.cpu_times()
    cpu_seconds = (cpu_end.user - cpu_start.user) + (cpu_end.system - cpu_start.system)
    cores = psutil.cpu_count()
    row = {
        "config": args.label or " ".join(args.set) or "padrão",
        "resolucao": f"{width}x{height}",
        "fps": round(len(totals) / wall, 1),
        "cpu_nucleos": round(cpu_seconds / wall, 2),           # núcleos ocupados em média
        "cpu_total_pct": round(100 * cpu_seconds / wall / cores, 1),  # como no Gerenciador de Tarefas
        "ms_frame_mediana": round(statistics.median(totals), 1),
        "ms_rosto": round(statistics.median(stages["rosto"]), 1),
        "ms_yolo_por_rodada": round(statistics.median(yolo_runs), 1) if yolo_runs else 0,
        "yolo_rodadas_por_s": round(len(yolo_runs) / wall, 1),
        "ms_regras": round(statistics.median(stages["regras"]), 2),
        "descartados": dropped,
    }

    print(f"\n== {row['config']} ({row['resolucao']}) ==")
    print(f"FPS processado: {row['fps']}  | descartados por atraso: {dropped}")
    print(f"CPU: {row['cpu_nucleos']} núcleos em média = {row['cpu_total_pct']}% da CPU ({cores} threads)")
    print(f"Tempo por frame (mediana): {row['ms_frame_mediana']} ms  | rosto {row['ms_rosto']} ms"
          f"  | YOLO {row['ms_yolo_por_rodada']} ms x {row['yolo_rodadas_por_s']}/s"
          f"  | regras {row['ms_regras']} ms")

    if args.csv:
        path = Path(args.csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not path.exists()
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            if new_file:
                writer.writeheader()
            writer.writerow(row)


if __name__ == "__main__":
    main()
