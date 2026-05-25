"""
Utilitarios para o projeto de Imitation Learning.
Classes e funcoes auxiliares.
"""

import os
import re
from pathlib import Path
import numpy as np
import torch as th
from stable_baselines3.common.policies import ActorCriticPolicy


def get_last_index(path: str, prefix: str, suffix: str = ".zip") -> int:
    """
    Encontra o maior numero de indice em arquivos dentro de um diretorio.
    
    Exemplo: Se existem 'bc_policy0.zip', 'bc_policy15.zip', retorna 15.
    
    Args:
        path: Caminho do diretorio
        prefix: Prefixo do arquivo (ex: "bc_policy")
        suffix: Sufixo do arquivo (ex: ".zip")
    
    Returns:
        Maior indice encontrado, ou -1 se nao houver arquivos
    """
    last_index = -1
    p = Path(path)
    if not p.exists():
        return last_index
    
    pattern = re.compile(re.escape(prefix) + r"(\d+)" + re.escape(suffix) + r"$")
    for file in p.iterdir():
        if file.is_file():
            m = pattern.search(file.name)
            if m:
                last_index = max(last_index, int(m.group(1)))
    return last_index


class LSTMWrapper:
    """
    Wrapper para politicas com extrator LSTM.
    Gerencia o estado do LSTM entre chamadas (reset entre episodios).
    """
    def __init__(self, policy: ActorCriticPolicy):
        self.policy = policy
        self.lstm_state = None
        self.feature_buffer = []  # Mantem buffer aqui
        self._last_obs = None
        
        # Verificar se a politica tem LSTM
        self.has_lstm = (hasattr(policy, 'features_extractor') and 
                        hasattr(policy.features_extractor, 'lstm'))
    
    def reset(self):
        """Reseta o estado LSTM (deve ser chamado a cada novo episodio)."""
        self.lstm_state = None
        self.feature_buffer = []
        # Chamar reset do features_extractor se existir
        if self.has_lstm and hasattr(self.policy, 'features_extractor') and \
           hasattr(self.policy.features_extractor, 'reset_hidden'):
            self.policy.features_extractor.reset_hidden()
    
    def predict(self, observation, deterministic: bool = False):
        """
        Faz predicao de acoes mantendo estado LSTM.
        
        Args:
            observation: Array com a observacao atual
            deterministic: Se True, sempre pega a acao mais provavel
        
        Returns:
            Acao predita e estado (None por compatibilidade)
        """
        with th.no_grad():
            # Converter para tensor se necessario
            if not isinstance(observation, th.Tensor):
                obs_tensor = th.as_tensor(observation).float()
            else:
                obs_tensor = observation.float()
            
            # Normalizar
            if obs_tensor.max() > 1.0:
                obs_tensor = obs_tensor / 255.0
            
            # Fazer predicao pela politica
            # Note: A politica do SB3 pode ter seu proprio gerenciamento de LSTM
            deterministic_float = 1.0 if deterministic else 0.0
            
            # Usar o metodo predict da politica base
            if hasattr(self.policy, 'predict'):
                action, state = self.policy.predict(observation, deterministic=deterministic)
                return action, state
    
    def __getattr__(self, name):
        """Delega acessos ao objeto de politica base."""
        return getattr(self.policy, name)
