# Imitation Player

**IA que aprende a jogar qualquer jogo vendo você jogar.**

Projeto de **Imitation Learning** para jogos de PC/emuladores. A IA observa a tela do jogo em tempo real, aprende a mapear "imagem da tela → ação do controle" usando uma rede neural CNN+LSTM+Attention, e depois reproduz o comportamento automaticamente.

---

## Visão Geral

```
  VOCÊ JOGA                IA APRENDE                 IA JOGA
     |                         |                          |
     v                         v                          v
+-----------+         +------------------+        +------------------+
|  Gravar   |   -->   | Behavioral Clone |  -->  |  Jogar sozinha   |
|  (.pt)    |         | (CNN+LSTM+Attn)  |        | (gamepad virt.)  |
+-----------+         +------------------+        +------------------+
                              |
                              v (opcional)
                    +-------------------+
                    | DAgger: você      |
                    | corrige a IA      |
                    +-------------------+
                              |
                              v
                    +-------------------+
                    | Retreinar com     |
                    | dados corrigidos  |
                    +-------------------+
```

---

## Estrutura do Projeto

```
imitation-player/
│
├── hajime_agent/                 # Agente focado em Hajime no Ippo (PS3/RPCS3)
│   ├── config/
│   │   └── game_config.py        # Configuração do jogo, ações, treinamento
│   ├── utils/
│   │   ├── game_env.py           # Ambiente Gymnasium (captura + controle virtual)
│   │   └── utils.py              # LSTMWrapper, get_last_index
│   ├── notebooks/
│   │   ├── record_trajectories.py  # Gravação de gameplay humana
│   │   ├── train_agent.py          # Treinamento BC com MLflow
│   │   ├── train_imiation.py       # Treinamento BC simplificado
│   │   ├── train_gail.py           # Treinamento GAIL (adversarial)
│   │   ├── run_ai.py               # Execução da IA no jogo
│   │   └── run_dagger.py           # Coleta interativa DAgger
│   ├── demos/                      # Trajetórias gravadas (.pt)
│   ├── models/                     # Modelos treinados (.zip)
│   ├── mlruns/                     # Logs do MLflow
│   └── GUIA_TREINAMENTO.md         # Guia passo a passo completo
│
├── generic_agent/                # Versão genérica (qualquer jogo/emulador)
│   ├── config/
│   │   └── game_config.py        # Exemplo configurado para Roblox (keyboard+mouse)
│   ├── utils/
│   │   ├── game_env.py           # Mesmo ambiente, suporta keyboard_mouse
│   │   └── utils.py              # Utilitários compartilhados
│   └── notebooks/                # Mesmos scripts do hajime_agent
│       ├── record_trajectories.py
│       ├── train_agent.py
│       ├── run_ai.py
│       └── ...
│
├── test_xinput.py                # Teste de detecção de controle XInput
├── requirements.txt              # Dependências exatas do projeto
├── Dockerfile                    # Imagem Docker para treino (Linux/NVIDIA)
└── .gitignore
```

---

## Tecnologias

| Componente | Tecnologia |
|---|---|
| Captura de tela | `dxcam` (GPU) com fallback `mss` |
| Controle virtual | `vgamepad` (Xbox 360 / DualShock 4) |
| Ambiente RL | `gymnasium` (Gym API) |
| Rede neural | `PyTorch` (CNN + LSTM bidirecional + Attention) |
| Algoritmos IL | `imitation` (BC, GAIL) |
| RL base | `stable-baselines3` (PPO, ActorCriticPolicy) |
| Tracking | `MLflow` |
| Visualização | `pygame` |

---

## Como Funciona

### 1. Ambiente de Jogo (`GenericGameEnv`)

Classe `gym.Env` que:
- **Encontra a janela do jogo** pelo nome do processo (ex: `rpcs3`, `re9.exe`)
- **Captura a tela** com `dxcam` (acelerado por GPU) em escala de cinza
- **Redimensiona** para 128x128 (entrada da rede neural)
- **Envia comandos** via `vgamepad` (controle Xbox 360 virtual) ou `pydirectinput` (teclado/mouse)
- Suporta **modo dummy** (treinamento sem jogo aberto)

### 2. Mapeamento de Ações (18 ações binárias)

| Índice | Ação | Controle |
|---|---|---|
| 0-3 | UP/DOWN/LEFT/RIGHT | D-Pad / Stick Esquerdo |
| 4 | A (Cross) | Face button |
| 5 | B (Circle) | Face button |
| 6 | X (Square) | Face button |
| 7 | L2 | Trigger esquerdo |
| 8 | R2 | Trigger direito |
| 9 | L3 | Apertar stick esquerdo |
| 10-13 | Câmera X | Stick direito (discretizado) |
| 14-17 | Câmera Y | Stick direito (discretizado) |

### 3. Arquitetura da Rede Neural (`TemporalAttentionLSTM`)

```
Imagem (128x128, grayscale)
    │
    ▼
┌─────────────────────────────────────┐
│  CNN (5x Conv2D + BatchNorm + ReLU) │
│  AdaptiveAvgPool → Flatten → 512    │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  LSTM Bidirecional (2 layers, 256)  │
│  Janela temporal de 10 frames       │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  Atenção Temporal (qual frame       │
│  é mais importante?)                │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  Linear(512 → 1024) → ReLU → Dropout│
│  Linear(1024 → 512) → ReLU          │
│           → Ação (18 bins)          │
└─────────────────────────────────────┘
```

- **CNN**: Extrai características espaciais (formas, bordas, texto, HUD)
- **LSTM**: Mantém memória dos últimos 10 frames (movimento, combos)
- **Attention**: Pondera quais frames são mais relevantes (o frame exato do soco importa mais que 2s de neutral)

### 4. Pipeline de Treinamento

**Fase 1 - Gravação** (`record_trajectories.py`):
- Jogue com o controle real (XInput) ou teclado
- A cada frame: tela + ações pressionadas são salvos como pares (obs, act)
- Pressione `K` para iniciar/parar gravação
- Trajetórias salvas em `demos/demo_*.pt`

**Fase 2 - Behavioral Cloning** (`train_agent.py`):
- A IA aprende por supervisão direta: dado o frame, qual ação o humano tomou?
- Suporta **transfer learning** (continuar treino de um modelo existente)
- Logs no **MLflow** (loss, acurácia, hiperparâmetros)

**Fase 3 - Teste** (`run_ai.py`):
- Carrega o modelo treinado
- IA joga sozinha via gamepad virtual
- Pressione `K` para alternar entre IA e manual
- Pressione `ESC` para sair

**Fase 4 - DAgger** (`run_dagger.py`):
- A IA joga, você observa e corrige os erros
- As correções viram novas trajetórias
- Retreina com dados antigos + novos
- Repita até a IA ficar boa

**Fase 5 - GAIL** (`train_gail.py`):
- Treinamento adversarial (gerador vs discriminador)
- Requer o jogo rodando (a IA precisa interagir com o ambiente real)
- Pode produzir políticas mais robustas que BC puro

---

## Como Usar

### Pré-requisitos

- **Python 3.11**
- **Windows** (para captura DX cam, win32gui, vgamepad)
- **GPU NVIDIA com CUDA** (recomendado para treino)
- **Jogo/emulador** aberto (RPCS3, PCSX2, Steam, etc.)

### Instalação

```powershell
# Clonar
git clone <repo>
cd imitation-player

# Ativar ambiente virtual
.\venv\Scripts\Activate.ps1

# Ou instalar do zero:
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Configuração

Edite `hajime_agent/config/game_config.py` (ou `generic_agent/config/game_config.py`):

```python
GAME_CONFIG = {
    "process_name": "rpcs3",         # Nome do processo (Task Manager)
    "exe_path": None,                 # Caminho do .exe (auto-abrir)
    "rom_path": None,                 # Caminho da ROM
    "capture": {
        "width": 854, "height": 480,  # Resolução da captura
        "internal_width": 128, "internal_height": 128,  # Resolução do modelo
        "target_fps": 60,
    },
    "actions": {
        "num_actions": 18,
        "input_mode": "gamepad",      # "gamepad" ou "keyboard_mouse"
    }
}
```

### Gravar Trajetórias

```powershell
cd hajime_agent/notebooks
python record_trajectories.py
```

- Abra o jogo antes
- Pressione `K` para gravar
- Pressione `ESC` para salvar e sair

### Treinar

```powershell
python train_agent.py --epochs 100 --batch 384 --lr 1e-4
```

Argumentos:
- `--epochs` : Épocas de treino (default: 100)
- `--batch` : Tamanho do batch (default: 384)
- `--lr` : Learning rate (default: 1e-4)
- `--device` : `cuda` ou `cpu`
- `--model_path` : Caminho para modelo pré-treinado (transfer learning)
- `--dagger` : Executar DAgger após treino

### Visualizar MLflow

```powershell
mlflow ui --backend-store-uri file:../mlruns
# Abra http://localhost:5000
```

### Testar a IA

```powershell
python run_ai.py
```

- `K` : Ativar/desativar IA
- `ESC` : Sair

### DAgger (Correção Humana)

```powershell
python run_dagger.py
```

- `K` : Gravar/parar
- `L` : Alternar entre IA e controle humano
- `ESC` : Sair

---

## Docker (Treino em Linux)

Como a captura de tela (`dxcam`) e controles virtuais (`vgamepad`) exigem Windows, a gravação e execução devem ser no host. O treinamento pode rodar em Docker com GPU.

```bash
# Construir
docker build -t imitation-player-train .

# Executar (montar diretórios de dados)
docker run --gpus all ^
  -v %cd%/hajime_agent/demos:/app/hajime_agent/demos ^
  -v %cd%/hajime_agent/models:/app/hajime_agent/models ^
  imitation-player-train
```

---

## Adaptação para Qualquer Jogo

### Jogos de Emulador (RPCS3, PCSX2, Dolphin)

1. Configure `process_name` para o nome do processo do emulador
2. Abra o emulador/jogo antes de rodar os scripts
3. O gamepad virtual Xbox 360 (`VX360Gamepad`) é reconhecido por todos os emuladores

### Jogos de PC (Steam, executável nativo)

1. Configure `process_name` para o nome do .exe
2. Se o jogo usa teclado/mouse, mude `input_mode` para `"keyboard_mouse"`
3. Mapeie as ações no `game_config.py`

### Exemplos de Configuração

| Jogo | `process_name` | `input_mode` |
|---|---|---|
| Hajime no Ippo (PS3/RPCS3) | `rpcs3` | `gamepad` |
| Resident Evil Requiem (PC) | `re9` | `gamepad` |
| Roblox (PC) | `RobloxPlayerBeta` | `keyboard_mouse` |
| PS2 (PCSX2) | `pcsx2-qt` | `gamepad` |

---

## Troubleshooting

| Problema | Solução |
|---|---|
| "Window not found" | Abra o jogo antes; verifique `process_name` |
| Captura lenta | Fallback para `mss` em laptops dual-GPU; reduza resolução |
| Gamepad não funciona | Teste com `test_xinput.py`; verifique se o driver ViGEm está instalado |
| IA só anda para os lados | Falta variedade nos dados; grave mais situações |
| IA aperta botões aleatórios | Treine mais épocas; verifique se a loss está descendo |
| GPU out of memory | Reduza `batch_size` |
| Overfitting | Grave mais dados diversos; aumente regularização |

---

## Licença

Baseado no trabalho de [@paulo101977](https://github.com/paulo101977/notebooks-rl).
