# Status do projeto

Atualizado em 24/11/2026. Para quem vai continuar o projeto (integrantes do grupo ou o Claude Code):
o que já foi feito, por que foi feito assim e o que vem a seguir.

## Onde estamos

As semanas 1, 2 e 3 (parte de sistema) estão prontas e testadas ao vivo, mais duas extensões (fadiga e
Telegram, ambas funcionando) e a otimização de custo. **Falta:** gravar e rotular os vídeos de teste (fim da
semana 3) e toda a semana 4 (avaliação, experimentos, relatório e apresentação).

## O que foi feito

| Etapa | Entregue |
|---|---|
| Semana 1 | Estrutura do repositório, ambiente, webcam, MediaPipe e YOLO funcionando (`scripts/check_setup.py`); código de referência executado (`scripts/run_reference.py`) |
| Semana 2 | Sonolência (EAR, calibração por pessoa, histerese, 2 s de olhos fechados); celular (confiança, filtros de tamanho/proporção/região, persistência de 1 s); HUD no estilo da referência |
| Extensão: fadiga | PERCLOS (60 s) e bocejos (MAR); estado FADIGA como aviso, separado da SONOLENCIA |
| Semana 3 (sistema) | Eventos com início/fim/duração, `logs/events.csv`, alerta sonoro, `src/pipeline.py` reutilizável |
| Extensão: Telegram | Mensagem com foto para a "central" no início de cada alerta (`scripts/telegram_setup.py` configura o bot). **Testado ao vivo em 24/11/2026:** a SONOLENCIA chegou no celular com foto (EAR 0,04 de olhos fechados, alerta em 2,0 s, 14,5 FPS) |
| Custo computacional | Webcam ao vivo: de 7,5 núcleos (47% da CPU) para 1,0 núcleo (6,1%), mesmo FPS. Tabela no README |

Testes: `python -m pytest` (47 testes, todos passando).

## Decisões e motivos

- **Python 3.13** (a proposta dizia 3.11): é o instalado e todas as bibliotecas funcionam.
- **MediaPipe Face Landmarker** (API Tasks) em vez do Face Mesh: a API antiga (`mp.solutions`) foi removida
  do MediaPipe. Os 478 pontos são os mesmos.
- **Limiar do EAR calibrado** (65% do EAR de olhos abertos, medido nos 3 s iniciais): o EAR de olho aberto
  medido foi ~0,36–0,40, bem acima do 0,20 fixo da referência. `--no-calibration` usa o valor fixo do
  `settings.yaml` (para os experimentos com 0,15/0,20/0,25).
- **Sonolência por tempo real** (2 s pelos timestamps), não por número de frames como na referência.
- **Rosto perdido**: mantém o estado do olho por 1 s (`face_lost_grace`), porque a cabeça pode cair ao cochilar.
- **MAR pelos lábios internos** (boca fechada ≈ 0; na foto de teste, boca entreaberta = 0,20); bocejo = MAR > 0,6 por 1 s.
- **FADIGA = PERCLOS ≥ 15% OU 3 bocejos em 5 min**, critérios transparentes em vez do índice ponderado da
  referência (pesos sem justificativa, e somava o celular, que é distração, à fadiga). O PERCLOS só é
  avaliado após 30 s de rosto visível.
- **Celular**: mantidos os filtros da referência (configuráveis) e acrescentada persistência de 1 s e região de interesse.
- **Telegram**: só no início do evento, no máximo 1 mensagem por tipo a cada 60 s; token em
  `config/secrets.yaml` (fora do git); erros nunca mostram o token.
- **Custo**: `OMP_WAIT_POLICY=PASSIVE` (maior ganho), limite de 15 FPS, YOLO a cada 0,2 s. Limitar threads do
  PyTorch e capturar em 640x480 não trouxeram ganho e não foram adotados.
- **Commits** só no nome dos integrantes, sem `Co-Authored-By`.

## Achados sobre o código de referência (para o relatório)

1. Não roda com o MediaPipe atual (`mp.solutions` foi removido); rodamos com um adaptador, sem alterar os arquivos dele.
2. O README diz que o YOLOv8n é quantizado, mas o `.onnx` está em FP32 (nenhum nó de quantização).
3. Uma única detecção de celular já dispara "PHONE USE" (não há persistência para disparar, só para manter).
4. Não aplica NMS: o mesmo celular pode gerar várias caixas.
5. O índice de fadiga usa pesos arbitrários (1,5 × PERCLOS + 15 por bocejo + distração).
6. Rodou a 16,8 FPS nesta máquina (YOLO a cada 2 frames).

## Problemas conhecidos e limitações

- **Celular de costas/inclinado**: o YOLOv8n não reconheceu um celular rosa de costas nem com confiança de 1%;
  o YOLO11s deu 0,34. O filtro de proporção também o descartaria (caixa quase quadrada). Avaliar com os vídeos;
  alternativas: YOLO11s, desligar o filtro de proporção ou fine-tuning.
- Se a cabeça cair e o rosto sumir por mais de 1 s, a contagem de sonolência zera.
- O "olho fechado" do PERCLOS é mais brando que o critério P80 da literatura; olhar para baixo também baixa o EAR.
- Bocejo coberto pela mão não é contado.

## Para quem clonar o projeto

1. Ambiente: `python -m venv .venv`, `pip install -r requirements.txt`, `python scripts/download_models.py`.
2. Telegram (opcional): o token **não está no git**. Peça o token do bot ao Pedro por mensagem privada, aperte
   "Iniciar" na conversa com o bot e rode `python scripts/telegram_setup.py` na sua máquina. Sem isso o programa
   funciona normalmente, só não manda mensagens.
3. Para testar alertas várias vezes seguidas: `python main.py --set telegram.cooldown_seconds=0` (o padrão é
   1 mensagem por tipo de alerta a cada 60 s).

## Próximos passos

1. **Vídeos de teste** (fim da semana 3): gravar em ambiente simulado, com consentimento, os cenários da
   proposta (normal, piscadas, olhos fechados por alguns segundos, celular visível, celular parcialmente oculto,
   mudança de iluminação). Rotular início e fim de cada evento (dois integrantes, conferência cruzada).
   **Próxima tarefa de código:** criar o script de gravação e definir o formato do arquivo de rótulos, já no
   formato que a avaliação vai ler.
2. **Semana 4 — avaliação**: script que roda `DriverMonitor` em cada vídeo (`--no-alerts`), compara os eventos
   com os rótulos e calcula precision, recall, F1, falsos alertas por minuto, latência do alerta e FPS.
3. **Experimentos da proposta**: limiar do EAR (0,15/0,20/0,25), tempo para alerta (1/2/3 s), confiança do YOLO
   (0,30/0,50/0,70), resolução (1280x720 e 640x480); incluir também `phone.imgsz` 320 e o filtro de proporção
   ligado/desligado. Usar `--set` para variar sem editar o `settings.yaml`.
4. Relatório, gráficos, vídeo demonstrativo e apresentação.
