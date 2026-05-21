# AI Game Agent - Imitation Learning para Qualquer Jogo

Projeto baseado no `re_requiem` de [@paulo101977](https://github.com/paulo101977/notebooks-rl), 
adaptado para ser generico e funcionar com qualquer jogo de PC ou emulador.

---

## O que este projeto faz?

Este projeto permite que voce crie uma **IA que aprende a jogar qualquer jogo vendo voce jogar**. 
A IA observa a tela do jogo e aprende a mapear "o que vejo" em "oque aperto no controle".

### Fluxograma do processo:

```
+------------------+    +-------------------+    +-------------------+
|  1. VOCE JOGA    | -> | 2. GRAVA TUDO     | -> | 3. IA APRENDE     |
|  (controle real) |    | (tela + botoes)   |    | (CNN+LSTM+Attn)   |
+------------------+    +-------------------+    +-------------------+
                                                           |
+------------------+    +-------------------+              v
|  6. CORRIGE/     | <- | 5. IA JOGA SOZIN. | <- | 4. VERIFICA       |
|     REFINA       |    | (gamepad virtual)   |    | (testa no jogo)   |
+------------------+    +-------------------+    +-------------------+
```

### Tecnologias:
- **CV**: `dxcam` (captura de tela acelerada por GPU)
- **Controle**: `vgamepad` (DualShock 4 virtual)
- **AI**: `stable-baselines3`, `imitacao` (Hugging Face)
- **LSTM**: Redes com memoria temporal para entender sequencias de frames
- **Atencao**: Foco automatico nos elementos relevantes da tela

---

## Estrutura do Projeto

```
hajime_agent/
    config/
        game_config.py          # Configure seu jogo aqui
    utils/
        game_env.py             # Ambiente generico (captura + controle)
        utils.py                # Funcoes auxiliares (LSTMWrapper, etc.)
    notebooks/
        record_trajectories.py  # Grava voce jogando
        train_imitation.py       # Treina a IA
        run_ai.py               # Faz a IA jogar sozinha
    demos/                      # Onde as trajetorias sao salvas
    models/                     # Onde os modelos treinados sao salvos
```

---

## Como Usar (Passo a Passo)

### 1. Instalacao

```bash
# Clone o repositorio
git clone https://github.com/paulo101977/notebooks-rl.git

# Instale as dependencias
pip install gymnasium stable-baselines3 numpy pygame dxcam imitacao vgamepad opencv-python torch psutil inputs keyboard

# Para GPU NVIDIA (opcional, mas RECOMENDADO)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. Configurar o Jogo

Edite `config/game_config.py`:

```python
GAME_CONFIG = {
    "process_name": "rpcs3",  # <-- nome do processo do emulador/jogo
    "exe_path": None,         # <-- deixe None se o jogo ja estiver aberto
    "capture": {
        "width": 854,
        "height": 480,
    },
    # ... restante
}
```

**Para emuladores:**
- **RPCS3 (PS3)**: `process_name="rpcs3"`, abra o jogo antes de iniciar
- **PCSX2 (PS2)**: `process_name="pcsx2-qt"`
- **Dolphin (Wii/GC)**: `process_name="dolphin"`
- **Jogo de PC**: nome do .exe (ex: "re9")

### 3. Gravar Trajetorias Humanas

Rode o script de gravacao:
```bash
cd notebooks
python record_trajectories.py
```

- Abra o emulador/jogo
- O script capturara a tela
- **Pressione `K`** para comecar/parar de gravar
- Jogue normalmente com o controle/teclado
- Grave pelo menos **10 trajetorias** (mais = melhor!)

As trajetorias sao salvas em `demos/demosX.pt`.

### 4. Treinar a IA

```bash
python train_imitation.py
```

- Ajuste `epochs`, `batch_size` e outras configs
- O modelo sera salvo em `models/bc_policy.zip`

### 5. Fazer a IA Jogar

```bash
python run_ai.py
```

- **Pressione `K`** para ativar/desativar a IA
- **Pressione `ESC`** para sair
- A IA controla o jogo sozinha via gamepad virtual!

### 6. (Opcional) Refinar com DAGGER

Execute o notebook `imitation-hg-dagger.ipynb` (ou adapte):
1. A IA joga
2. Voce corrige quando ela erra
3. Gera mais trajetorias corrigidas
4. Retreina com dados novos
5. Repita ate ficar boa!

---

## Adaptando para seu Jogo Especifico

### Mapeamento de Acoes

No `game_env.py`, edite a funcao `step` para mapear as acoes corretamente.

Exemplo para **Hajime no Ippo (PS3)** - jogo de boxe:
```
Acoes necessarias:
- Esquerda/Direita (movimento ring)
- Cima/Baixo (esquivas/bloqueios)  
- X (jab rapido esquerda)
- Circulo (direto direita)
- Quadrado (gancho esquerda)
- Triangulo (uppercut)
- L2 (bloquear)
- R2 (especial/fervor)
- L3 (movimento de pes)
- Stick Direito: camera/movimento corpo
```

### Dicas para Jogos de Luta (como Hajime no Ippo):

1. **Timing e Frame Data**: Atraso de 50-100ms pode ser critico para combos frame-perfect. 
   Ajuste `target_fps` para capturar mais frames por segundo.

2. **Diversidade**: Grave contra varios oponentes em diferentes estagios.
   A IA precisa ver muitas situacoes diferentes para generalizar.

3. **Teoria de Jogos**: Jogos de luta sao adversariais (o oponente reage as sua acoes).
   DAGGER e essencial, pois a IA inicial so joga bem contra os dados de treino.

4. **Observabilidade**: A IA so ve a tela. Se ela precisar saber a vida de ambos 
   os lutadores, isso deve estar visivel na tela (pas de HUD separado).

---

## Arquitetura da Rede Neural

```
Imagem (128x128x1) -->
[Conv2D] -- [BatchNorm] -- [ReLU] x5 -->
[Flatten] -- (512 features) -->
[LSTM Bidirectional] -- (cap. temporal: ultimos 10 frames) -->
[Atencao Temporal] -- (foca nos frames mais importantes) -->
[Linear 1024] -- [ReLU] -- [Dropout] -->
[Linear 512] --> Acao predita (18 botoes)
```

### Por que Atencao + LSTM?

- **LSTM**: Permite a IA lembrar o que aconteceu nos ultimos 10 frames (ex: sequencia de combo,
  posicao anterior no ring, movimento do oponente).

- **Atencao Tempora**: O jogo nem todos os frames sao igualmente importantes. Ex: o frame 
  exato do contra-ataque e mais importante que 2 segundos de neutral. A atencao da peso maior
  para os frames mais relevantes.

---

## Guia para Hajime no Ippo (PS3 via RPCS3)

### Passo 1: Configuracao
1. Instale o RPCS3 e o jogo
2. Configure o controle no RPCS3 (XInput/DirectInput)
3. Ajuste `config/game_config.py`:
   ```python
   GAME_CONFIG = {
       "process_name": "rpcs3",
       "exe_path": None,  # Abra o RPCS3 manualmente
       "capture": {
           "width": 1280,   # Resolucao interna do RPCS3
           "height": 720,
       },
   }
   ```

### Passo 2: Mapeamento de Botoes
No `game_env.py`, `step()`:
```python
# Mapeamento para Hajime no Ippo
# X = jab esquerda
# O = direto direita  
# [] = gancho esquerda
# Tri = uppercut
# L2 = bloquear
# R2 = fervor/especial
```

### Passo 3: Gravacao
1. Abra o jogo no modo Carreira ou Amistoso
2. Jogue contra um oponente especifico (para consistencia inicial)
3. Faca movimentos basicos:
   - Movimentar ring
   - Esquivas
   - Combos basicos (Jab -> Direto -> Gancho)
   - Bloqueios e Contra-Ataques
4. Grave pelo menos 15-20 trajetorias de 1 minuto cada

### Passo 4: Treinamento
- Inicial: 50 epochs (geral)
- Dagger 1: +30 epochs (comecar a corrigir)
- Dagger 2: +30 epochs (refinar)
- Dagger 3+: +20 epochs (polir)

### Passo 5: Teste
- Veja a IA jogar
- Anote os momentos que ela erra
- Use o notebook `dagger` para corrigir esses momentos
- Retreine e repita

---

## Solucao de Problemas

| Problema | Solucao |
|----------|---------|
| "Window not found" | Abra o jogo ANTES de rodar o script; verifique o nome do processo |
| Captura lenta/feia | Se for emulador, reduza a resolucao interna; deixe o jogo em tela cheia minimizado |
| Delay na IA | Reduza `target_fps` para sincronizar; ou use GPU melhor |
| IA so anda para os lados | Falta dados! Grave mais trajetorias com movimentos variados |
| IA aperta botoes aleatorios | Treine por mais epochs; verifique se a loss esta decrescendo |
| Gamepad nao funciona | Verifique se vgamepad esta instalado corretamente; teste gamepad virtual |

---

## Licenca

Baseado no trabalho de [paulo101977](https://github.com/paulo101977/notebooks-rl).
Este projeto foi adaptado para tornar o uso mais generico e acessivel.

