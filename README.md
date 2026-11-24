# Sistema de Monitoramento de Motorista com Visão Computacional

Protótipo acadêmico que monitora um motorista por uma câmera comum e detecta, em tempo real,
**sonolência** (olhos fechados por tempo prolongado) e **uso de celular**, com MediaPipe e YOLOv8n.

PUC Minas, Campus Poços de Caldas · Visão Computacional e Realidade Misturada · Prof. Will Machado
Integrantes: Pedro Henrique Franco Moreira Amaral, Artur Carlo Costa Padua, Enzo Batista Salerno, João Henrique de Oliveira

> Protótipo acadêmico, testado só em ambiente simulado. Não é um sistema de segurança automotiva certificado.

## Andamento

- [x] **Semana 1:** estrutura do repositório, ambiente, webcam, landmarks e YOLO funcionando, código de referência executado
- [x] **Semana 2:** detector de olhos e sonolência (EAR, calibração, histerese, temporizador); detector de celular (limiar, filtros, região, persistência); testes isolados
- [x] **Extensão:** fadiga (PERCLOS e bocejos), como aviso antecipado separado da sonolência
- [x] **Semana 3 (sistema):** motor de decisão com eventos (início, fim, duração), interface, alerta sonoro, `events.csv`, notificação no Telegram (extensão)
- [ ] **Semana 3 (dados):** gravação e rotulagem dos vídeos de teste
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
python main.py                      # monitor completo na webcam
python main.py --source video.mp4   # num arquivo de vídeo
python main.py --no-calibration     # usa o limiar fixo do EAR (para os experimentos)
python main.py --no-alerts          # sem som e sem Telegram
python -m pytest                    # testes unitários
```

Teclas: `q` ou `ESC` sai, `c` recalibra os olhos, `m` mostra a malha completa do rosto.

Ao abrir, o sistema passa 3 s **calibrando**: olhe para a câmera com os olhos abertos. Ele mede o seu EAR
de olhos abertos e usa 65% dele como limiar de olho fechado. Depois disso:

- **SONOLENCIA**: olhos fechados (EAR abaixo do limiar) de forma contínua por 2 s. Piscadas não disparam.
- **CELULAR DETECTADO**: celular aceito pelos filtros por pelo menos 1 s. Falhas de até 0,5 s não zeram a contagem.
- **FADIGA** (aviso, faixa laranja): cansaço acumulado. Liga com **PERCLOS ≥ 15%** (fração do tempo com olhos
  fechados no último minuto, avaliada após 30 s de rosto visível) ou com **3 bocejos em 5 min** (boca muito
  aberta, MAR > 0,6, por pelo menos 1 s; a fala não chega a isso).
- Caixa **amarela** = celular visto, mas ainda não confirmado; **vermelha** = confirmado; **cinza** = descartado pelos filtros.

### Saídas

- **Som:** bipe contínuo enquanto durar a SONOLENCIA (para acordar o motorista); dois bipes curtos no início
  do CELULAR; um bipe grave no início da FADIGA.
- **`logs/events.csv`:** uma linha por evento, acrescentada a cada execução. Colunas: `sessao`, `fonte`,
  `evento`, `inicio_s`, `fim_s`, `duracao_s` (tempo do vídeo, para comparar com os rótulos), `valor`
  (EAR mínimo na SONOLENCIA, confiança máxima no CELULAR, PERCLOS máximo na FADIGA), `detalhe` e
  `data_hora_inicio`. O início de cada evento é o instante do alerta.
- **Telegram (extensão):** mensagem com foto para a "central" quando um alerta começa, com no máximo uma
  mensagem a cada 60 s por tipo de alerta.

### Configurar o Telegram

1. No Telegram, abra o **@BotFather**, mande `/newbot`, escolha nome e username (terminando em `bot`) e copie o token.
2. Abra a conversa com o seu bot e aperte **Iniciar**. Para alertar um grupo ("central"), adicione o bot ao
   grupo e mande `/start@nome_do_bot` lá.
3. Rode `python scripts/telegram_setup.py` e cole o token. O script descobre o chat, salva tudo em
   `config/secrets.yaml` (fora do git) e manda uma mensagem de teste.

A foto enviada mostra o rosto do motorista: use só com o consentimento de quem aparece no vídeo.

`scripts/check_setup.py` é a verificação da semana 1 (sem regras temporais) e, ao sair, imprime o FPS médio e
a latência de cada modelo.

Todos os parâmetros (fonte de vídeo, resolução, limiares) ficam em [config/settings.yaml](config/settings.yaml).

## Estrutura

```
main.py                   programa principal
config/settings.yaml      parâmetros e limiares
config/secrets.yaml       token do Telegram (criado pelo telegram_setup.py, fora do git)
src/pipeline.py           processamento de um frame: detectores → regras → decisão → eventos
src/capture/              captura de webcam ou vídeo, com timestamp por frame
src/detectors/            face_landmarks + eyes (EAR, calibração, sonolência),
                          mouth (MAR, bocejo) + fatigue (PERCLOS, fadiga),
                          phone_yolo + phone (filtros, persistência), temporal (regra de duração)
src/decision/             motor de decisão (lista de estados ativos) e eventos (início/fim)
src/interface/            HUD, alerta sonoro, events.csv e Telegram
src/utils/                configuração e FPS
scripts/                  download dos modelos, verificação, execução da referência
tests/                    testes unitários (pytest)
models/                   pesos baixados (fora do git)
```

Os timestamps vêm do relógio na webcam e da posição no vídeo em arquivos. Assim as regras temporais
(ex.: olhos fechados por 2 s) independem do FPS, e um vídeo de teste dá o mesmo resultado em qualquer computador.

## Referências técnicas

- SOUKUPOVÁ, T.; ČECH, J. *Real-Time Eye Blink Detection using Facial Landmarks*. CVWW, 2016 (EAR).
- DINGES, D. F.; GRACE, R. *PERCLOS: A Valid Psychophysiological Measure of Alertness as Assessed by
  Psychomotor Vigilance*. FHWA, 1998 (PERCLOS). Aqui o "olho fechado" usa o mesmo limiar calibrado da
  sonolência, mais brando que o critério P80 da literatura; por isso o limiar de 15% deve ser ajustado nos experimentos.

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
