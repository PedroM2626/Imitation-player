# Imitation Player - Hajime no Ippo

Este projeto implementa um agente de inteligência artificial que aprende a jogar **Hajime no Ippo - The Fighting! (PS3)** via aprendizado por imitação (Behavioral Cloning e DAgger), utilizando capturas de tela aceleradas por GPU (via `dxcam`) e comandos virtuais (via `vgamepad`).

---

## Estrutura do Workspace

- `hajime_agent/`: Diretório do agente contendo as configurações de captura, mapeamento de botões e modelos de IA.
- `venv/`: Ambiente virtual Python 3.11.9 do projeto com todas as dependências instaladas.
- `requirements.txt`: Dependências exatas do projeto.
- `Dockerfile`: Especificação para execução do ambiente de treino em contêineres Docker.

---

## Configuração do Sistema

As configurações principais do sistema do usuário foram unificadas no arquivo central de configurações: [game_config.py](file:///d:/Imitation-player/hajime_agent/config/game_config.py).

Configurações ativas:
- **Executável do Emulador**: `D:\emuladores\rpcs3-v0.0.39-18737-818b11fd_win64_msvc\rpcs3.exe`
- **Caminho da ROM**: `D:\roms\Hajime no Ippo - The Fighting! (Japan).iso`
- **Ambiente Virtual**: Configurado em `venv/` usando Python 3.11.9.

---

## Como Gravar as Trajetórias Humana

Siga o passo a passo abaixo para gravar suas gameplays:

1. **Configurar o RPCS3**:
   - Abra o RPCS3 manualmente pela primeira vez.
   - Certifique-se de configurar as opções de controle (Pads). O mapeamento deve apontar para o gamepad virtual gerado pelo `vgamepad` (normalmente reconhecido como controle de Xbox/DualShock pelo Windows).

2. **Iniciar o Gravador**:
   - Abra um terminal no diretório raiz do projeto e ative o ambiente virtual:
     ```powershell
     .\venv\Scripts\Activate.ps1
     ```
   - Navegue para o diretório de notebooks e execute o script de gravação:
     ```powershell
     cd hajime_agent/notebooks
     python record_trajectories.py
     ```
   - O emulador RPCS3 será iniciado automaticamente e carregará a ROM do jogo caso ele ainda não esteja aberto.

3. **Gravar a Gameplay**:
   - O script exibirá uma tela de visualização (preview).
   - Coloque o jogo na tela de gameplay ativo (luta).
   - **Pressione 'K'** no seu teclado para iniciar a gravação. O status mudará para **RECORDING**.
   - Jogue utilizando seu controle convencional normalmente. Foque em movimentos variados (ataques, esquivas, bloqueios e combos).
   - **Pressione 'K'** novamente para pausar a gravação de uma trajetória. Cada trajetória gravada será contabilizada na tela de preview.
   - Repita o processo até completar o número mínimo de trajetórias (10-20 sugeridas).
   - Ao fechar ou concluir as gravações, as trajetórias serão salvas automaticamente como arquivos `.pt` no diretório `hajime_agent/demos/`.

---

## Treinamento e MLOps com MLflow

O pipeline de treinamento está totalmente integrado com os princípios de MLOps utilizando o **MLflow**. Toda execução de treino gerará uma run com logs de hiperparâmetros, estatísticas dos datasets, perdas por época (`loss`) e o arquivo final do modelo.

### Executando o Treinamento

1. Execute o script de treinamento:
   ```powershell
   python train_agent.py --epochs 100 --batch 384 --lr 1e-4
   ```

2. Visualizando experimentos no MLflow:
   - Para abrir o painel de visualização do MLflow, execute:
     ```powershell
     mlflow ui --backend-store-uri file:../mlruns
     ```
   - Abra o navegador no endereço `http://localhost:5000` para acompanhar as métricas, comparar execuções e baixar modelos gerados.

---

## Execução via Docker (Treinamento)

Como a captura de tela (`dxcam`) e os controles virtuais (`vgamepad`) requerem integração direta com o kernel do Windows, a gravação e a execução de jogo autônomo devem ser feitas no host Windows. Contudo, o treinamento dos modelos (que requer alto processamento gráfico) pode ser empacotado e rodado em qualquer ambiente Linux/NVIDIA Docker.

### Construir a Imagem de Treino
```bash
docker build -t imitation-player-train .
```

### Executar o Treino no Docker (com suporte à GPU)
```bash
docker run --gpus all -v $(pwd)/hajime_agent/demos:/app/hajime_agent/demos -v $(pwd)/hajime_agent/models:/app/hajime_agent/models imitation-player-train
```

---

## Como Testar o Agente

Após concluir o treinamento e ter o arquivo `bc_policy.zip` salvo no diretório `models/`, siga os passos abaixo para testar o agente no jogo:

1. **Preparar o Emulador e o Jogo**:
   - Abra o RPCS3 e inicie o jogo **Hajime no Ippo - The Fighting!**.
   - Navegue no jogo até entrar em uma luta ativa (onde o personagem possa se mover e lutar).

2. **Ativar o Ambiente Virtual e Rodar o Teste**:
   - Abra um terminal no diretório raiz do projeto e execute:
     ```powershell
     .\venv\Scripts\Activate.ps1
     cd hajime_agent/notebooks
     python run_ai.py
     ```
   - O script procurará a janela ativa do emulador (`rpcs3`).
   - Se o script falhar ao inicializar o `dxcam` (comum em laptops híbridos/dual-GPU), ele fará o **fallback automático para a biblioteca `mss`** e funcionará normalmente.

3. **Interagir com o Agente de IA**:
   - Uma janela de visualização do Pygame será aberta, exibindo a tela redimensionada que a IA vê (128x128 ampliada para a janela 854x480).
   - **Ativar/Desativar IA (Tecla 'K')**: Pressione a tecla **'K'** no teclado a qualquer momento para ligar ou desligar o controle da IA. 
     - Quando ativa, o indicador no topo esquerdo ficará verde com o status **AI ACTIVE**, e a IA assumirá o controle do personagem.
     - Quando inativa, o status será **MANUAL**, e você poderá controlar manualmente o personagem ou apenas observar o jogo livremente.
   - **Sair do Teste (Tecla 'Esc')**: Pressione a tecla **'Esc'** a qualquer momento para encerrar o script de execução e fechar a visualização do Pygame.

