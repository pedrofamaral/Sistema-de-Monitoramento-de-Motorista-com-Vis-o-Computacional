# CLAUDE.md

Contexto para o Claude Code neste repositório. O andamento, as decisões e os próximos passos estão em
@docs/STATUS.md — leia antes de começar e atualize ao terminar uma etapa.

## Projeto

Protótipo acadêmico (PUC Minas, Visão Computacional e Realidade Misturada, prof. Will Machado) que monitora
um motorista por webcam e detecta **SONOLENCIA**, **CELULAR DETECTADO** e **FADIGA**. O escopo segue a
proposta de 4 semanas do grupo; extensões além dela: fadiga (PERCLOS + bocejos) e notificação no Telegram.

## Regras do projeto

- Converse em **português**. Nomes no código em inglês; comentários, docstrings, README e textos em português.
  Textos desenhados com OpenCV ficam **sem acento** (a fonte Hershey não tem acentos).
- **Commits**: mensagem em português, autor = integrante do grupo. **Nunca** adicione `Co-Authored-By` nem
  qualquer atribuição ao Claude em commits ou PRs.
- **Nenhum limiar fixo no código**: todo parâmetro vai em `config/settings.yaml`.
- **Tempo sempre pelos timestamps dos frames**, nunca por contagem de frames, para o resultado não depender
  do FPS e um vídeo de teste dar o mesmo resultado em qualquer computador.
- **Custo computacional importa**: meça com `scripts/benchmark.py` antes e depois de mudanças no pipeline.
  Não remova o `OMP_WAIT_POLICY=PASSIVE` de `src/__init__.py` (sem ele o PyTorch ocupa ~7 núcleos à toa).
- **Nunca commite** `config/secrets.yaml` (token do Telegram), `models/`, `data/videos/` nem `logs/`.
- Ao terminar uma etapa: `python -m pytest`, atualizar `docs/STATUS.md` e o `README.md`.

## Comandos (Windows, PowerShell)

```powershell
python -m venv .venv; .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/download_models.py          # face_landmarker.task e yolov8n.pt em models/
python main.py                             # monitor na webcam (q sai, c recalibra, m malha)
python main.py --source video.mp4 --no-alerts --set drowsiness.closed_seconds=3
python -m pytest                           # testes unitários
python scripts/benchmark.py --source 0 --seconds 30   # custo de CPU ao vivo
python scripts/telegram_setup.py           # liga o bot do Telegram
```

## Arquitetura

`main.py` lê frames de `src/capture/` e chama `src/pipeline.py` (`DriverMonitor.process`), que faz por frame:
MediaPipe Face Landmarker → EAR (`detectors/eyes.py`: calibração, histerese, 2 s) e MAR
(`detectors/mouth.py`: bocejos) → PERCLOS (`detectors/fatigue.py`); YOLOv8n a cada 0,2 s
(`detectors/phone_yolo.py`) → filtros e persistência (`detectors/phone.py`); regra de duração genérica em
`detectors/temporal.py`; `decision/engine.py` devolve a lista de estados e `decision/events.py` transforma em
eventos com início e fim. Saídas em `src/interface/`: HUD, som, `logs/events.csv` e Telegram.
O mesmo `DriverMonitor` deve ser usado na avaliação dos vídeos de teste (semana 4).
