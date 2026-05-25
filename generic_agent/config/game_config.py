"""
Configuracao Generica para Qualquer Jogo
Ajuste estas variaveis de acordo com o jogo que voce quer treinar.
"""

# ==============================================
# CONFIGURACAO DO JOGO (ALTERE AQUI)
# ==============================================

GAME_CONFIG = {
    # Processo do jogo ou emulador
    # Exemplos: "rpcs3" (PS3), "pcsx2-qt" (PS2), "dolphin" (Wii/GameCube), 
    #           "re9" (Resident Evil Requiem), "hajime_no_ippo" (exemplo)
    "process_name": "RobloxPlayerBeta",
    
    # Caminho para o executavel do jogo (opcional - deixe None se ja estiver aberto)
    # Exemplo: R"C:\Games\RPCS3\rpcs3.exe"
    # Ou para Steam: R"C:\Program Files (x86)\Steam\steam.exe -applaunch APPID"
    "exe_path": None,
    
    # Caminho para a ROM do jogo
    "rom_path": None,
    
    # Configuracoes da captura de tela
    "capture": {
        # Ajuste-vos conforme a resolucao em que o jogo sera jogado
        # O modelo usa 128x128 internamente, mas essa e a resolucao bruta capturada
        "width": 854,       # Largura da captura (ex: 854 para 480p, 1280 para 720p)
        "height": 480,      # Altura da captura
        "internal_width": 128,    # Resolucao que o modelo processa
        "internal_height": 128,   # Resolucao que o modelo processa
        "target_fps": 60,       # FPS alvo da captura
        "buffer_len": 1,         # Buffer do DXCam (1 = mais rapido, 3 = mais suave)
    },
    
    # Offset da janela (ajuste se necessario para evitar bordas/menu do emulador)
    "window_offset": {
        "left": 20,     # pixels a esquerda do rect da janela
        "top": 100,     # pixels do topo (pular a barra de titulo, etc.)
        "right": 0,     # pixels a direita
        "bottom": 0,    # pixels de baixo
    },
    
    # ===========================================
    # MAPEAMENTO DE ACOES (ALTERE CONFORME O JOGO)
    # ===========================================
    # Ajuste as acoes de acordo com o seu jogo.
    # O tamanho da lista define o numero de acoes.
    # Cada acao e binaria (0 ou 1).
    "actions": {
        # Numero de acoes (default: 9). Altere aqui se necessario.
        "num_actions": 9,
        
        # Define se a IA vai usar "gamepad" (controle Xbox) ou "keyboard_mouse" (teclado e mouse)
        "input_mode": "keyboard_mouse", 
        
        # Mapeamento dos botoes. Para adicionar mais, basta criar uma nova linha e aumentar o num_actions.
        "mappings": [
            # Teclado WASD
            {"name": "W", "type": "key", "key": "w"},
            {"name": "A", "type": "key", "key": "a"},
            {"name": "S", "type": "key", "key": "s"},
            {"name": "D", "type": "key", "key": "d"},
            
            # Teclas extras
            {"name": "SPACE", "type": "key", "key": "space"},
            {"name": "F", "type": "key", "key": "f"},
            {"name": "R", "type": "key", "key": "r"},
            
            # Botoes do Mouse
            {"name": "CLICK_L", "type": "mouse_button", "button": "left"},
            {"name": "CLICK_R", "type": "mouse_button", "button": "right"},
        ]
    }
}


# ==============================================
# CONFIGURACAO DO TREINAMENTO
# ==============================================

TRAINING_CONFIG = {
    # numero maximo de trajetorias por sessao de gravacao (limite baixo para evitar MemoryError ao salvar)
    "max_trajectories": 3,
    
    # tamanho do batch para treinamento
    "batch_size": 384,
    
    # numero de epocas de treinamento
    "epochs": 100,
    
    # learning rate
    "learning_rate": 1e-4,
    
    # tamanho do buffer temporal (LSTM window)
    "window_size": 10,
    
    # numero de passos DAGGER
    "dagger_iterations": 3,
    
    # diretorios
    "demo_path": "./demos/",
    "model_path": "./models/",
    "train_path": "./models/imitation/",
}


# ==============================================
# CONFIGURACAO DO CONTROLE (INPUT)
# ==============================================

INPUT_CONFIG = {
    # Deadzone para sticks analogicos (0-1)
    "deadzone": 0.3,  # 30% do movimento maximo
    
    # sensibilidade da camera (multiplicador do stick direito)
    "camera_sensitivity": 1.0,
    
    # Delay entre leituras do gamepad (segundos)
    "input_delay": 0.01,
    
    # FATOR DE AGRESSIVIDADE (Multiplicador de probabilidade para botoes do mouse)
    # 1.0 = Normal (A IA so bate se tiver certeza)
    # 2.0 = Agressivo (A IA bate com metade da certeza necessaria)
    # 3.0 = Muito Agressivo (A IA spamma ataques na menor intencao)
    "aggressiveness": 2.0,
}
