"""Sistema de monitoramento de motorista com visão computacional."""
import os

# Custo computacional: por padrão, as threads do OpenMP usadas pelo PyTorch (YOLO)
# ficam "girando" enquanto esperam a próxima inferência e ocupam ~7 núcleos o tempo
# todo, mesmo com o YOLO rodando só a cada 0,2 s. PASSIVE faz elas dormirem na espera
# (medido: 7 -> 0,5 núcleo). Precisa ser definido antes de o torch ser importado,
# por isso fica aqui, no primeiro módulo carregado. Quem já definiu a variável no
# ambiente (ex.: OMP_WAIT_POLICY=ACTIVE para comparar) mantém o próprio valor.
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
