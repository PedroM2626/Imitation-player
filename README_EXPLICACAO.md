# Projeto `re_requiem` - Explicacao Completa

Analise do projeto de treinar um agente de IA para jogar **Resident Evil Requiem (RE9)** no PC usando Imitacao de Aprendizado (Behavioral Cloning/DAGGER).

---

## O que o projeto faz?

O projeto cria um **agente de IA que joga Resident Evil Requiem (PC/Steam)**.
A IA aprende assistindo um humano jogar e depois reproduzindo as acoes. O processo basico e:

1. **Enxergar o jogo**: Captura a tela do jogo em tempo real.
2. **Gravar o humano jogando**: Salva as acoes (botoes pressionados) alinhadas com o que aparecia na tela.
3. **Treinar a IA**: A IA aprende a mapear "imagem da tela -> acao" usando uma rede neural com LSTM (memoria temporal).
4. **Rodar a IA no jogo**: A IA controla o jogo sozinha, observando a tela e apertando botoes virtualmente.

---

## Arquivos do Projeto

| Arquivo | Funcao |
|---------|--------|
| `resident_requiem.py` | Ambiente do jogo (Gymnasium) que conecta com o jogo real, captura tela e envia comandos.
| `imitatation-record.ipynb` | Grava trajetorias (gameplay humano) usando gamepad real.
| `trainning-imitation.ipynb` | Treina a rede neural usando as trajetorias gravadas (Behavioral Cloning / DAGGER).
| `running-imitation-lstm.ipynb` | Executa a IA treinada no jogo para testar a performance.
| `imitatation-hg-dagger.ipynb` | Faz coleta de dados DAGGER: a IA joga e o humano corrige suas acoes, gerando novos dados para treinamento. |

---

## Hardware e Software que voce precisa

### Hardware
- **GPU NVIDIA com CUDA**: Para acelerar o treinamento da rede neural.
- **Controle de Xbox/PlayStation**: Para gravar os movimentos humanos e para correcao DAGGER.

### Software necessario
- Python 3.11 (o codigo usa `imp` que foi removido no Python 3.12+)
- Windows (o codigo usa `win32gui`, `win32process`, e gamepad virtual DS4)
- Jogo **Resident Evil Requiem** instalado no Steam (ou outro jogo que voce quer adaptar)

### Bibliotecas Python chave
```bash
pip install gymnasium stable-baselines3 numpy pygame dxcam imitacao vgamepad opencv-python torch psutil inputs keyboard
```

---

## Como funciona cada parte?

### 1. `resident_requiem.py` - O Ambiente do Jogo

Este arquivo define a classe `RERequiemEnv`, que:

- **Encontra a janela do jogo** pelo nome do processo (`re9.exe`).
- **Captura a tela** em tempo real usando `dxcam` (captura de tela acelerada por GPU).
- **Envia comandos** ao jogo usando `vgamepad` (gamepad virtual DualShock 4).
- **Define o espaco de acoes**: Sao 18 acoes binarias (botoes do controle).
- **Observacoes**: A tela do jogo em RGB (480x854x3), resuzida para 128x128 para o modelo.

#### Mapeamento de Acoes (18 indices)
| Indice | Acao |
|--------|------|
| 0 | UP (D-pad ou Stick Esquerdo Y+) |
| 1 | DOWN (D-pad ou Stick Esquerdo Y-) |
| 2 | LEFT (D-pad ou Stick Esquerdo X-) |
| 3 | RIGHT (D-pad ou Stick Esquerdo X+) |
| 4 | CROSS (X / A) |
| 5 | CIRCLE (O / B) |
| 6 | SQUARE (□ / Y) |
| 7 | L2 Trigger |
| 8 | R2 Trigger |
| 9 | L3 (pressionar Stick Esquerdo) |
| 10-13 | Stick Direito X (discretizado: -0.5, -1, +0.5, +1) |
| 14-17 | Stick Direito Y (discretizado: -0.5, -1, +0.5, +1) |

#### Extrator de Features: LSTM Temporal
A parte mais avancada do codigo e a classe `TemporalAttentionLSTM`:

- **CNN**: Extrai features espaciais da imagem (5 camadas convolucionais).
- **LSTM+Atencao**: Mantem uma janela de 10 frames passados para entender movimento/sequencia temporal.
- **Rede final**: Mapeia os features para 512 dimensoes (usado pela politica do Stable Baselines3).

Isso permite que a IA lembre o que aconteceu nos ultimos 10 frames, essencial para jogos de terc pessoa com movimento e camera.

---

### 2. `imitatation-record.ipynb` - Gravacao Humanas

Este notebook:

1. **Abre a janela do jogo** e captura frames em tempo real.
2. **Le o gamepad real** via `inputs` (biblioteca que le eventos de controle USB).
3. **Converte os botoes do controle** para o formato de acoes binarias (18 dimensoes).
4. **Envia as acoes para o jogo** via o `RERequiemEnv` (controla o personagem via gamepad virtual).
5. **Grava pares (observacao, acao)** em trajetorias.
6. **Salva as trajetorias** em arquivos `.pt` (pickle do PyTorch).

**Obs**: O gamepad real le as entradas, e o `RERequiemEnv` reenvia via gamepad virtual para o jogo. Isso garante sincronia perfeita entre o que o humano faz e o que a IA enxerga.

---

### 3. `trainning-imitation.ipynb` - Treinamento

Este notebook treina a IA usando as trajetorias gravadas. O metodo principal e **Behavioral Cloning (BC)** da biblioteca `imitacao` (Hugging Face + Farama).

#### Algoritmo DAGGER (Dataset Aggregation)
O projeto usa uma variante de **DAGGER**:
1. Comeca com trajetorias humanas.
2. Treina a politica inicial com BC.
3. Coloca a IA para jogar (`imitacao-hg-dagger`).
4. O humano corrige as acoes da IA quando ela erra.
5. Adiciona essas trajetorias corrigidas ao dataset.
6. Retreina a IA com os dados novos.
7. Repete ate a IA ficar boa.

#### Configuracoes de treinamento (exemplos)
- **Batch size**: 384
- **Epocas**: Multiplas passadas sobre os dados
- **Entropia**: Regularizacao para evitar confianca excessiva (overconfident)
- **Loss**: Neglogp (cross-entropy) + entropia + L2 regularizacao

---

### 4. `running-imitation-lstm.ipynb` - Rodar a IA

Este notebook carrega o modelo treinado e faz a IA jogar sozinha.

Funciona assim:
1. Carrega a politica `ActorCriticPolicy` do ultimo checkpoint (`bc_politica{epoch}.zip`).
2. Envolve com `LSTMWrapper` (para manter o estado LSTM).
3. Em loop:
   - Captura a tela atual.
   - Passa pela rede neural.
   - Obtem a acao predita (stochastic ou deterministica).
   - Envia para o jogo via `RERequiemEnv`.

Tem modo manual tambem (use o teclado) para intervir a qualquer momento.

---

### 5. `imitation-hg-dagger.ipynb` - Coleta DAGGER

Este notebook permite que o humano corrija a IA em tempo real:

1. A IA joga (prediz acoes baseadas na tela).
2. Se a IA errar, o humano aperta botaoes no controle e a IA aprende com a correcao.
3. Isso gera novas trajetorias que sao salvas e usadas no proximo treinamento.

Tem modos:
- **deterministico**: A IA joga com confianca maxima (para testes).
- **stochastic**: A IA explora mais (para variedade de trajetorias).

---

## Como adaptar para Hajime no Ippo (PS3) ou Qualquer Outro Jogo

Para usar este projeto em outro jogo (como **Hajime no Ippo** no PS3, ou qualquer jogo de PC/emulador), voce precisa alterar principalmente o arquivo `resident_requiem.py`.

### Passo 1: Alterar o processo alvo
```python
# No __init__ de RERequiemEnv
self.hwnd = self.find_window_by_process_name("re9")  # <--- ALTERAR AQUI
```

Para PS3 via RPCS3:
```python
self.hwnd = self.find_window_by_process_name("rpcs3")
```

Para um emulador de PS2 ou outro:
```python
self.hwnd = self.find_window_by_process_name("pcsx2-qt")
```

Para Steam (se o nome da janela for diferente):
```python
# Use o nome da janela/exe do jogo
self.hwnd = self.find_window_by_process_name("hajime_no_ippo")
```

### Passo 2: Adicionar deteccao do emulador
O RPCS3 e um emulador de PS3. Para controlar o jogo dentro do RPCS3 via gamepad virtual, o gamepad virtual deve funcionar como um controle conectado ao Windows (o que `vgamepad` faz). A maioria dos emuladores reconhece gamepads XInput/DirectInput automaticamente.

Voce pode precisar mapear os botoes de acordo com o layout do controle do PS3 (X, Circulo, Quadrado, etc).

### Passo 3: Ajustar o caminho do executavel (opcional)
Se quiser que o script lance o jogo automaticamente:
```python
process = subprocess.Popen([
    r"C:\Caminho\Para\rpcs3.exe", 
    r"--no-gui", 
    r"C:\Jogos\HajimeNoIppo\PS3_GAME\USRDIR\EBOOT.BIN"
])
```

### Passo 4: Ajustar o mapeamento de acoes
O projeto usa 18 acoes. Voce pode precisar de menos ou mais botoes para o seu jogo.

Para um jogo de luta como Hajime no Ippo, o mapeamento pode ser:
- Direcionais: 4 acoes (cima, baixo, esquerda, direita)
- Botoes de acao: 4 acoes (X, Circulo, Quadrado, Triangulo)
- Bloco/Desvio: 1 acao
- Special: 1 acao

Total: ~10-16 acoes (depende do jogo).

Altere a classe `RERequiemEnv` para mapear corretamente os botoes do controle virtual para os botoes do jogo.

### Passo 5: Ajustar a resolucao da captura
Pode ser que o jogo/emulador tenha resolucao diferente:
```python
self.height = 480  # ou 720, 1080, etc.
self.width = 640   # ajuste conforme a tela do emulador
```

### Passo 6: Rodar o pipeline completo
1. Grave trajetorias com `imitation-record.ipynb`.
2. Treine com `trainning-imitation.ipynb`.
3. Teste com `running-imitation-lstm.ipynb`.
4. Colete correcoes DAGGER com `imitation-hg-dagger.ipynb`.
5. Retreine.
6. Repita ate a IA ficar boa.

---

## Fluxograma do Pipeline Completo

```
+-----------------+     +-----------------+     +------------------+
|  HUMANO JOGA    | --> |  GRAVA DADOS   | --> | TREINA IA (BC)   |
|  (gamepad real) |     |  (obs, acoes)  |     |  (LSTM + CNN)    |
+-----------------+     +-----------------+     +------------------+
         ^                                              |
         |                                              v
+--------+--------+                           +------------------+
|  HUMANO CORRIGE |                           |   IA JOGA SOZINHA|
|  (DAGGER loop)  | <-- Coleta --             |  (gamepad virtual)|
+-----------------+                           +------------------+
         |                                              |
         +---------------- TREINA NOVAMENTE ------------+
```

---

## Como adaptar genericamente para qualquer jogo

Para criar um projeto generico, voce pode transformar o `resident_requiem.py` em um arquivo de configuracao separada.+ Exemplo de refatoracao:

### `config.py` (criar novo arquivo)
```python
# Configuracao generica para qualquer jogo
GAME_CONFIG = {
    "process_name": "rpcs3",           # nome do processo do jogo/emulador
    "exe_path": None,                   # None se ja estiver aberto
    "actions": {
        "num_actions": 18,
        "mapping": {
            "UP": 0,
            "DOWN": 1,
            # ... defina os mapeamentos de acoes
        }
    },
    "capture": {
        "width": 854,
        "height": 480,
        "target_fps": 240
    }
}
```

Depois, modifique `RERequiemEnv` para aceitar essa config:
```python
class GenericGameEnv(gym.Env):
    def __init__(self, config):
        self.config = config
        self.hwnd = self.find_window_by_process_name(config["process_name"])
        # ... restante usando self.config
```

---

## Dicas importantes

1. **Performance de captura**: O `dxcam` e muito mais rapido que `mss` ou `PIL.ImageGrab`. Se o RPCS3 lagar, tente reduzir a resolucao interna do emulador.

2. **Gamepad virtual**: O `vgamepad` cria um DualShock 4 virtual. Alguns emuladores preferem XInput (Xbox). Nesse caso, use `vg.VX360Gamepad()` em vez de `vg.VDS4Gamepad()`.

3. **Delay (atraso)**: Existe delay natural entre captura da tela e envio de comando. Para jogos de luta (frame-perfect), o delay pode ser um problema. Ajuste o `FRAME_TIME` e teste.

4. **Variabilidade dos dados**: Para jogos de luta, grave MUITAS trajetorias com diferentes oponentes, estagios e combinacoes de botoes. A IA precisa ver diversidade para generalizar.

5. **Melhorar a coleta DAGGER**: No modo DAGGER, quando a IA errar, o humano deve corrigir RAPIDAMENTE (em menos de 1 segundo), pois a IA aprende com a sequencia completa de acoes.

6. **Emulador de PS3 (RPCS3)**: Verifique se o gamepad virtual e reconhecido. No RPCS3, va em Pads > Pad Settings e verifique se aparece "XInput" ou o gamepad virtual.

