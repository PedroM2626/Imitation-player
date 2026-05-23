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
    "process_name": "rpcs3",
    
    # Caminho para o executavel do jogo (opcional - deixe None se ja estiver aberto)
    # Exemplo: R"C:\Games\RPCS3\rpcs3.exe"
    # Ou para Steam: R"C:\Program Files (x86)\Steam\steam.exe -applaunch APPID"
    "exe_path": R"D:\emuladores\rpcs3-v0.0.39-18737-818b11fd_win64_msvc\rpcs3.exe",
    
    # Caminho para a ROM do jogo
    "rom_path": R"D:\roms\Hajime no Ippo - The Fighting! (Japan).iso",
    
    # Configuracoes da captura de tela
    "capture": {
        # Ajuste-vos conforme a resolucao em que o jogo sera jogado
        # O modelo usa 128x128 internamente, mas essa e a resolucao bruta capturada
        "width": 854,       # Largura da captura (ex: 854 para 480p, 1280 para 720p)
        "height": 480,      # Altura da captura
        "internal_width": 64,    # Resolucao que o modelo processa (64x64 para evitar MemoryError)
        "internal_height": 64,   # Resolucao que o modelo processa (64x64 para evitar MemoryError)
        "target_fps": 240,       # FPS alvo da captura
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
        # Numero de acoes (default: 18). Altere aqui se necessario.
        "num_actions": 18,
        
        # Mapeamento dos botoes para o gamepad virtual
        # Tipos disponiveis: "button" (botao), "axis" (stick analogico), "trigger" (gatilho)
        "mappings": [
            # Exemplo: D-Pad / Stick Esquerdo
            {"name": "UP", "type": "axis", "vg_code": "left_joystick", "axis": "y", "value": -1.0},
            {"name": "DOWN", "type": "axis", "vg_code": "left_joystick", "axis": "y", "value": 1.0},
            {"name": "LEFT", "type": "axis", "vg_code": "left_joystick", "axis": "x", "value": -1.0},
            {"name": "RIGHT", "type": "axis", "vg_code": "left_joystick", "axis": "x", "value": 1.0},
            
            # Botoes de face
            {"name": "CROSS", "type": "button", "vg_code": "DS4_BUTTON_CROSS"},      # X no PlayStation
            {"name": "CIRCLE", "type": "button", "vg_code": "DS4_BUTTON_CIRCLE"},    # O no PlayStation
            {"name": "SQUARE", "type": "button", "vg_code": "DS4_BUTTON_SQUARE"},     # □ no PlayStation
            
            # Gatinhos (para jogos de luta, pode ser desvio/parry)
            {"name": "L2", "type": "trigger", "vg_code": "left_trigger", "value": 255},   # Gatilho esquerdo
            {"name": "R2", "type": "trigger", "vg_code": "right_trigger", "value": 255},  # Gatilho direito
            
            # L3 (pressionar stick esquerdo) - pode ser dash/run
            {"name": "L3", "type": "button", "vg_code": "DS4_BUTTON_THUMB_LEFT"},
            
            # Stick Direito X (Camera / desvio) - discretizado em 4 direcoes
            {"name": "CAM_RIGHT", "type": "axis", "vg_code": "right_joystick", "axis": "x", "value": 0.5},
            {"name": "CAM_RIGHT_FAST", "type": "axis", "vg_code": "right_joystick", "axis": "x", "value": 1.0},
            {"name": "CAM_LEFT", "type": "axis", "vg_code": "right_joystick", "axis": "x", "value": -0.5},
            {"name": "CAM_LEFT_FAST", "type": "axis", "vg_code": "right_joystick", "axis": "x", "value": -1.0},
            
            # Stick Direito Y
            {"name": "CAM_UP", "type": "axis", "vg_code": "right_joystick", "axis": "y", "value": -0.5},
            {"name": "CAM_UP_FAST", "type": "axis", "vg_code": "right_joystick", "axis": "y", "value": -1.0},
            {"name": "CAM_DOWN", "type": "axis", "vg_code": "right_joystick", "axis": "y", "value": 0.5},
            {"name": "CAM_DOWN_FAST", "type": "axis", "vg_code": "right_joystick", "axis": "y", "value": 1.0},
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
}
