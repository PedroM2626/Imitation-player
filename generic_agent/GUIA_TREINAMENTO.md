# Como Treinar o Agente - Guia Passo a Passo

## Visao Geral do Processo de Treinamento

```
   VOCE JOGA                  IA APRENDE                  IA JOGA
      |                             |                           |
      v                             v                           v
+-----------+              +----------------+          +-------------------+
|  Gravar   |     -->      |  Treinar com   |  -->   |   Testar no       |
|  dados    |              |  Behavioral    |        |   jogo real       |
|  (.pt)    |              |  Cloning       |        |   (run_ai.py)     |
+-----------+              +----------------+        +-------------------+
                                        |
                                        v (opcional)
                              +-------------------+
                              |   DAgger: voce    |
                              |   corrige a IA    |
                              |   juntando dados  |
                              +-------------------+
                                        |
                                        v
                              +-------------------+
                              |  Retreinar com    |
                              |  dados novos      |
                              +-------------------+
```

---

## FLUXO DE TREINAMENTO COMPLETO

### PRE-REQUISITOS

1. **Instale as dependencias**:
```bash
# Abra o terminal/prompt

# Instale PyTorch (versao para sua GPU ou CPU)
# Para GPU NVIDIA:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Para CPU:
pip install torch torchvision torchaudio

# Demais bibliotecas
pip install stable-baselines3 gymnasium
pip install imitation pillow numpy dxcam vgamepad
pip install keyboard pygame opencv-python psutil
```

2. **Tenha GPU NVIDIA com CUDA** (opcional mas recomendado):
   - O treinamento em GPU e 10-50x mais rapido que em CPU
   - Verifique: python -c "import torch; print(torch.cuda.is_available())"

---

## FASE 1: COLETA DE DADOS (Gravacao)

### Objetivo
Grave voce mesmo jogando para que a IA aprenda "copiando" suas acoes.

### Passo 1.1: Configure o jogo

Edite `config/game_config.py`:

```python
GAME_CONFIG = {
    "process_name": "rpcs3",  # <- ALTERE! Nome do executavel/emulador
    "exe_path": None,  # <- ou: caminho do .exe se quiser auto-abrir
}
```

**Para Hajime no Ippo (PS3 via RPCS3):**
- Open o emulador RPCS3 e o jogo ANTES de gravar
- `process_name="rpcs3"` funciona para encontrar a janela

### Passo 1.2: Abra o emulador/jogo

Abra o game/emulator e deixe na tela do gameplay, nao nos menus.

### Passo 1.3: Rode o gravador

```bash
cd notebooks
python record_trajectories.py
```

**Interface:**
- Tela de preview mostrando o que esta sendo capturado
- **Pressione 'K'** para INICIAR/PAUSAR a gravacao
- Grave em blocos de gameplay variado (diferentes situacoes)

### Passo 1.4: Grave pelo menos 10-20 trajetorias

Cada trajetoria deve durar 30 segundos a 2 minutos. Gravando:
- Movimentacao basica
- Combos simples
- Situacoes de defesa/evasao
- Diferentes oponentes/estagios (se possivel)

**Dica de Ouro: Variedade e mais importante que quantidade!**
Grave diferentes angulos de camera, movimentos, e situacoes de luta. A IA precisa ver como voce
age em muitas situacoes diferentes.

**Arquivos serao salvos em:** `demos/demos0.pt`, `demos1.pt`, etc.

---

## FASE 2: VERIFICACAO DOS DADOS

Antes de treinar, confira que os dados estao bons:

```bash
python -c "
import torch
from pathlib import Path

files = list(Path('./demos').glob('demos*.pt'))
print(f'Arquivos: {len(files)}')

for f in files:
    data = torch.load(f)
    print(f'{f.name}: {len(data)} trajetorias')
    if len(data) > 0:
        for i, traj in enumerate(data):
            print(f'  Traj {i}: {len(traj.obs)} frames, {traj.acts.shape} acoes')
"
```

---

## FASE 3: TREINAMENTO INICIAL (Behavioral Cloning)

### Comando Basico

```bash
cd notebooks
python train_agent.py --epochs 100 --batch 384 --lr 1e-4
```

### Parametros:

| Parametro     | Significado                          | Valor Padrao | Recomendado            |
|---------------|--------------------------------------|-------------|-----------------------|
| `--epochs`    | Numero de passadas pelos dados       | 100         | 50-300 (mais dados=dai) |
| `--batch`     | Imagens processadas por vez          | 384         | 256-512               |
| `--lr`        | Velocidade de aprendizado            | 1e-4        | 1e-5 a 1e-3           |
| `--device`    | 'cuda' ou 'cpu'                      | auto        | Se CUDA disponivel     |

### O que acontece durante o treinamento:

```
Episode 1/100
  Batch 0/50: loss=4.21, accuracy=12.5%
  Batch 1/50: loss=3.85, accuracy=15.2%
  ...
  Batch 49/50: loss=1.42, accuracy=51.3%
Episode 2/100
  ...
  Batch 49/50: loss=0.98, accuracy=68.7%
```

### Como saber que esta indo bem?

**Indicadores positivos:**
- **Loss diminuindo**: Comeca em ~4-5, deve chegar abaixo de 1.0
- **Acuracia crescendo**: A % de previsoes corretas aumenta
- **Batch completo sem erros**: Sem crashes ou overflows

**Quando parar (early stopping):**
- Se a loss "plateau" (para de descer) por 10+ epocas
- Se acuracia chegar a >90% (rapido demais, pode estar memorizando)
- Se validacao (>20 epocas) piora (overfitting)

---

## FASE 4: TESTAR A IA

### Teste simples

Abra outro terminal e execute:

```bash
cd notebooks
python run_ai.py
```

**Controles durante o teste:**
- **K**: Ativar/desativar IA
- **ESC**: Sair

### O que observar no teste:

- [ ] A IA esta parada demais?
- [ ] Ela faz movimentos repetitivos (loop)?
- [ ] Ela "congela" em situacoes especificas?
- [ ] Ela toca no jogo (parece estar viva)?

**Se sim:** Voce precisa de DAGGER (fase 5)

---

## FASE 5: REFINAMENTO (DAGGER)

DAGGER (Dataset Aggregation) e essencial para jogos complexos. Voce avalia:

```
Muitas situacoes: Layer2      |  Layer1: IA treinada
a IA treinada (base) nao vai com |  IA Base (BC)
ser perfeita de primeira. Nas     |    |
situacoes onde ela erra, voce      |  +--- Correio Humano
apenas corrige:                    |    |        ^
                                   |    |        |
+----------+   +-----------+       |  Nova Trajet.
| IA joga  |-->| Voce     |       |    +-------->
| Sozinha  |   | Corrige  |
+----------+   +-----------+
     ^               |
     |               v
+----------+   +-----------+
| Nova IA  |<--| Retreinada|
| Melhor   |   | com dados |
+----------+   +-----------+
```

### Como fazer DAGGER:

1. Execute `imitation-hg-dagger.ipynb` (ou crie um script)
2. A IA joga automaticamente
3. Você observa os erros
4. Quando a IA erra, voce PEGA O CONTROLE e faz a acao correta
5. Salva as trajetorias de CORRECAO em `demos/dagger/`
6. **Combinar dados:**
   ```bash
   # Copie todos os demos para pasta unificada
   python -c "
   from pathlib import Path
   import shutil
   Path('./demos_combined').mkdir(exist_ok=True)
   for f in Path('./demos').glob('demos*.pt'):
       shutil.copy(f, './demos_combined/')
   for f in Path('./demos/dagger').glob('demos*.pt'):
       shutil.copy(f, './demos_combined/')
   print('Dados combinados!')
   "
   ```
7. **Retreinar** com todos os dados:
   ```bash
   python train_agent.py --epochs 100 --batch 384
   ```
8. **Testar novamente** e repetir!

---

## FASE 6: TREINAMENTO AVANCADO (Opcional)

### Ajustes finos para performance:

**1. Regularizacao:**
Se IA estiver "overfitting" (memorizar dados em vez de aprender):

```python
# Na classe Trainer()
BC(
    observation_space=...,
    action_space=...,
    rng=...,
    batch_size=384,
    ent_weight=1e-3,    # Penalidade de entropia (exploracao)
    l2_weight=0.0,      # Regularizacao L2
)
```

**2. Transfer Learning:**
```python
# Congele camadas menores durante o treino inicial para
# treinar mais rapido (se os dados tem algum padrao repetitivo)
```

**3. Data Augmentation:**
Considere adicionar ruido leve nas imagens, mudancas de brilho, etc.

**4. Reduce Overfitting:**
- Gravando dados diversos (caminhos diferentes)
- Aumentando o numero de trajetorias
- Diminuindo o numero de parametros aumentando a regularizacao

---

## COMANDOS RAPIDOS (Cheat Sheet)

```bash
# ===== GRAVAR =====
# Precisa: jogo aberto na tela
python record_tragetorias.py        # Grava trajetorias humanas

# ===== VERIFICAR =====
python -c "from pathlib import Path; import torch; \
  [print(f'{f.name}: {len(torch.load(f))} traj.') for f in Path('./demos').glob('*.pt')]"

# ===== TREINAR (Rapido) =====
python train_agent.py --epochs 50 --batch 256

# ===== TREINAR (Padrao) =====
python train_agent.py --epochs 100 --batch 384

# ===== TREINAR (Completo+DAGGER) =====
python train_agent.py --epochs 200 --batch 512 --dagger

# ===== TESTAR =====
python run_ai.py
```

---

## PROBLEMAS COMUNS E SOLUCOES

| Problema                            | Causa Provavel                      | Solucao                                                                                                                                                                                                         |
|-------------------------------------|-------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| "No data found"                     | Pasta vazia ou nome errado          | Verifique que demos/ existe e tem .pt, confirme a variavel `demo_path` no script                                                                                                                               |
| Loss muito alta (nao diminui)       | Dados ruins ou LR muito alto        | Gravacao reiniciou; ajuste LR; verifique consistencia dos dados (acoes correspondem as telas?)                                                                                                                 |
| Loss=0 imediatamente                | Bug nos dados                       | Verifique se `traj.acts` esta vazio ou NaN; verifique shapes das observacoes (devem ser np.array ou array compativel)                                                                                          |
| IA faz sempre a mesma coisa         | Pouca variacao nos dados/OR         | Grave mais trajetorias variadas/DAGGER; dados nao balanceados (acoes muito frequentes); adicionar peso diferente para acoes raras                                                                              |
| IA trava/nao faz nada               | Bug no ambiente                     | Verifique se `game_env.py` esta recebendo apenas observacoes processadas corretamente (tipo uint8) quando skip de frame se processa para cuidadoso                                                        |
| GPU out of memory                   | Batch muito grande                   | Diminua `batch_size`; usa mixed precision training (apenas futuro); monitor uso da VRAM com `nvidia-smi` no terminal                                                                                            |
| "CUDA not available"                 | Driver da NVIDIA nao instalado       | Instale drivers NVIDIA; verifica CUDA Toolkit; ou use `--device cpu` (treino sera beeem mais lento)                                                                                                              |
| IA toca musicas/menu em vez do jogo | `process_name` errado                | Confirme o nome do processo que contem a janela do game realmente (verifica no Task Manager/Monitor de recursos), alguns emulators precisam do **nome da janela** em vez do nome do processo em alguns S.O. |

---

## EXEMPLO COMPLETO (Linha de Comando)

```bash
# 1. Instalar dependencias
pip install torch stable-baselines3 gymnasium imitacao pygame dxcam vgamepad

# 2. Gravar trajetorias (abrir RPCS3 + game primeiro)
python record_trajectories.py  # Pressione 'K' para gravar, faca 15 traj.

# 3. Verificar dados
ls demos/  # Confira demos0.pt, demos1.pt...
python -c "import torch; print(len(torch.load('demos/demos0.pt')))"

# 4. Treinar
python train_agent.py --epochs 100

# 5. Testar
python run_ai.py  # Pressione 'K' para ativar IA

# 6. (Opcional) DAgger para corrigir erros
# - Abra hacker_dagger.ipynb (ou implementation linguage-neutral)
# - Rode iterative sessions of corrections

# 7. Retreinar com dados DAGGER juntos
python train_agent.py --epochs 150 --lr 5e-5

# 8. Testar novamente
python run_ai.py
```

---

## MELHORES PRATICAS

1. **Grave em sessoes curtas (1-2 min)**: 
   Facilita o treinamento e permite maior diversidade.

2. **Variedade antes de quantidade**:
   20 traj. com movimentos variados > 100 traj. fazendo sempre a mesma coisa.

3. **Dados Balanceados**:
   Se voce jogar apenas ofensivamente a IA nunca aprendera a defender.

4. **Monitore a Loss**
   Se 'loss' comecar a crescer apos muito treino -> overfitting. Use Early Stopping.

5. **Dagger Iterativa**:
   Nao espere a IA ficar perfeita de uma vez. DAGGER funciona em:
   - Retreinar com ~30-50 novos exemplos por iteracao
   - 5-8 iteracoes DAGGER melhoram drasticamente

6. **Experimente Hiperparametros**:
   `learning_rate`: 1e-5 (lento mas preciso) vs 1e-3 (rapido mas pode"saltar")
   `batch_size`: 128 (pouca GPU) vs 512 (GPU rapida)

