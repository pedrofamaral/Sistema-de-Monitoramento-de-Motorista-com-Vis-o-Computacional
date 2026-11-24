# Sistema de Monitoramento de Motorista com Visão Computacional

Protótipo acadêmico que monitora um motorista por uma câmera comum e detecta, em tempo real,
**sonolência** (olhos fechados por tempo prolongado) e **uso de celular**, com MediaPipe e YOLOv8n.

PUC Minas, Campus Poços de Caldas · Visão Computacional e Realidade Misturada · Prof. Will Machado
Integrantes: Pedro Henrique Franco Moreira Amaral, Artur Carlo Costa Padua, Enzo Batista Salerno, João Henrique de Oliveira

> Protótipo acadêmico, testado só em ambiente simulado. Não é um sistema de segurança automotiva certificado.

## Andamento

- [x] **Semana 1:** estrutura do repositório, ambiente, webcam, landmarks e YOLO funcionando, código de referência executado
- [ ] **Semana 2:** detector de olhos e sonolência (EAR + temporizador); detector de celular (limiar, região, persistência)
- [ ] **Semana 3:** motor de decisão, interface, `events.csv`, gravação dos vídeos de teste
- [ ] **Semana 4:** experimentos, métricas, análise de erros, documentação e apresentação

## Instalação

Requer Python 3.11 ou mais recente (desenvolvido com 3.13) e uma webcam.

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
python scripts/download_models.py # baixa face_landmarker.task e yolov8n.pt para models/
```

## Como rodar

```bash
python scripts/check_setup.py                     # webcam: olhos, celular e FPS na tela
python scripts/check_setup.py --source video.mp4  # mesmo teste num arquivo de vídeo
python -m pytest                                  # testes unitários
```

Na janela: `q` ou `ESC` sai, `m` mostra a malha completa de landmarks. Ao sair, o script imprime o FPS
médio e a latência de cada modelo.

Todos os parâmetros (fonte de vídeo, resolução, limiares) ficam em [config/settings.yaml](config/settings.yaml).

## Estrutura

```
config/settings.yaml      parâmetros e limiares
src/capture/              captura de webcam ou vídeo, com timestamp por frame
src/detectors/            MediaPipe Face Landmarker (olhos) e YOLO (celular)
src/decision/             motor de decisão (semana 3)
src/interface/            HUD, alertas e events.csv (semana 3)
src/utils/                configuração e FPS
scripts/                  download dos modelos, verificação, execução da referência
tests/                    testes unitários (pytest)
models/                   pesos baixados (fora do git)
```

Os timestamps vêm do relógio na webcam e da posição no vídeo em arquivos. Assim as regras temporais
(ex.: olhos fechados por 2 s) independem do FPS, e um vídeo de teste dá o mesmo resultado em qualquer computador.

## Código de referência

[AreebaAminn/Driver-Drowsiness-Distraction-AlertEdge](https://github.com/AreebaAminn/Driver-Drowsiness-Distraction-AlertEdge) (MIT).
Ele usa `mp.solutions.face_mesh`, API removida nas versões atuais do MediaPipe. O script
`scripts/run_reference.py` roda o `main.py` original com um adaptador sobre o Face Landmarker, sem alterar
os arquivos da referência:

```bash
git clone https://github.com/AreebaAminn/Driver-Drowsiness-Distraction-AlertEdge referencia/AlertEdge
pip install onnxruntime pygame
python scripts/run_reference.py
```
