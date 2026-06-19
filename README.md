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

### 3. Arquiteturas de Rede Neural Disponiveis

O projeto agora suporta três arquiteturas diferentes para processar as imagens e tomar decisões. Você pode comparar todas usando o script `compare_models.py`.

#### A. NatureCNN (Padrão)
A arquitetura clássica do Stable-Baselines3. É uma Rede Neural Convolucional simples e rápida (3 camadas).
- **Vantagem**: Treina muito rápido, modelo muito leve.
- **Desvantagem**: Não tem memória temporal explícita, reage apenas ao frame atual (apesar de receber 4 frames empilhados).

#### B. CNN + LSTM + Attention (`train_agent_lstm.py`)
```
Imagem (128x128) → CNN (5 layers) → LSTM Bidirecional → Temporal Attention → Ação
```
- **CNN**: Extrai características espaciais (formas, bordas, texto, HUD)
- **LSTM**: Mantém memória dos últimos 10 frames (movimento, combos)
- **Attention**: Aprende a focar nos frames mais importantes da sequência antes de tomar a decisão final.

#### C. Vision Transformer (ViT) (`train_agent_transformer.py`)
```
Imagem (4x 128x128) → Patch Embedding (64 patches/frame) → Temporal Embedding → Transformer Encoder (4 layers) → Ação
```
Substitui a CNN inteira por um mecanismo de Self-Attention puro.
- A imagem é dividida em blocos (patches) de 16x16.
- A rede aprende relações espaciais e temporais simultaneamente prestando atenção em todos os blocos de todos os 4 frames ao mesmo tempo.
- **Vantagem**: Pode capturar padrões complexos globais na tela.
- **Desvantagem**: Mais pesada e demora mais para treinar.

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

## Guia Completo de Treinamento (Passo a Passo)

### Fase 1: Coleta de Dados (Gravação)

```powershell
cd hajime_agent/notebooks
python record_trajectories.py
```
- Abra o jogo/emulador antes.
- Pressione `K` para iniciar/pausar a gravação e `ESC` para salvar.
- **Dica de Ouro:** Variedade é mais importante que quantidade! Grave 15-20 trajetórias curtas (1-2 min) com diferentes movimentos, oponentes e cenários.

### Fase 2: Treinamento Inicial (Behavioral Cloning)

```powershell
python train_agent.py --epochs 100 --batch 384 --lr 1e-4
```
**Como saber que está indo bem?**
- **Loss diminuindo**: Começa em ~4-5, deve chegar abaixo de 1.0.
- Se a loss "congelar" por muitas épocas ou a acurácia passar de 90% muito rápido, pode ser *overfitting*.
- Para acompanhar os gráficos em tempo real, use o MLflow:
```powershell
mlflow ui --backend-store-uri file:../mlruns
# Abra http://localhost:5000
```

### Fase 3: Testar a IA

```powershell
python run_ai.py
```
- Pressione `K` para alternar entre IA e controle humano.
- Observe: A IA está muito parada? Faz movimentos repetitivos? Fica travada no canto? Se sim, você precisa da Fase 4.

### Fase 4: Refinamento (DAgger)

DAgger (Dataset Aggregation) é a técnica secreta para IA perfeita. A IA treinada nunca será excelente de primeira. Nas situações onde ela erra, você apenas assume o controle e corrige o erro no ato.

```
+----------+   +-----------+
| IA joga  |-->| Você      |
| Sozinha  |   | Corrige   |
+----------+   +-----------+
     ^               |
     |               v
+----------+   +-----------+
| Nova IA  |<--| Retreinada|
| Melhor   |   | com dados |
+----------+   +-----------+
```
```powershell
python run_dagger.py
```
- Pressione `L` para alternar a força entre IA e Humano.
- Corrija os erros, salve e **Retreine** (`python train_agent.py`) com todos os dados juntos.
- 5 a 8 iterações de DAgger costumam deixar a IA formidável.

---

## Melhores Práticas de Machine Learning

1. **Grave em sessões curtas**: Facilita o treinamento e permite maior diversidade.
2. **Dados Balanceados**: Se você jogar apenas ofensivamente, a IA nunca aprenderá a defender ou recuar. Mostre a ela os dois lados.
3. **Experimente Hiperparâmetros**: Se a IA não melhora, teste um `learning_rate` menor (`1e-5` é lento mas preciso) ou um `batch_size` diferente (use `512` se tiver muita VRAM).

---

## Troubleshooting

| Problema | Solução |
|---|---|
| "Window not found" | Abra o jogo antes; verifique `process_name` |
| Captura lenta | Fallback para `mss` em laptops dual-GPU; reduza resolução |
| Gamepad não funciona | Teste com `test_xinput.py`; verifique se o driver ViGEm está instalado |
| IA só anda para os lados | Falta variedade nos dados; grave mais situações balanceadas |
| Loss muito alta (não diminui) | Dados sujos ou LR muito alto. Tente regravar dados mais consistentes |
| GPU out of memory | Reduza o `batch_size` |

---

## Licença

Baseado no trabalho de [@paulo101977](https://github.com/paulo101977/notebooks-rl).
