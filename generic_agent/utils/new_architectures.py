"""
Novas Arquiteturas de Extratores de Características para Imitation Learning.
Inclui:
- ImpalaCNNExtractor (Impala original com Flatten)
- ResNet18Extractor (ResNet-18 adaptada para 4 canais)
"""

import torch as th
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torchvision.models as models

# ============================================================
# 1. IMPALA CNN (Original com Flatten)
# ============================================================

class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()

    def forward(self, x: th.Tensor) -> th.Tensor:
        out = self.relu1(x)
        out = self.conv1(out)
        out = self.relu2(out)
        out = self.conv2(out)
        return x + out


class ImpalaBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.res1 = ResidualBlock(out_channels)
        self.res2 = ResidualBlock(out_channels)

    def forward(self, x: th.Tensor) -> th.Tensor:
        x = self.conv(x)
        x = self.maxpool(x)
        x = self.res1(x)
        x = self.res2(x)
        return x


class ImpalaCNNExtractor(BaseFeaturesExtractor):
    """
    Impala CNN Extractor com Flatten (arquitetura padrão do artigo IMPALA).
    Achata a saída convolucional total, preservando informações espaciais detalhadas.
    """
    def __init__(self, 
                 observation_space: gym.spaces.Box,
                 features_dim: int = 512,
                 channels_list = [32, 64, 128]):
        super().__init__(observation_space, features_dim)
        
        self.n_frames = observation_space.shape[0]
        self.img_h = observation_space.shape[1]
        self.img_w = observation_space.shape[2]
        
        blocks = []
        in_ch = self.n_frames
        for out_ch in channels_list:
            blocks.append(ImpalaBlock(in_ch, out_ch))
            in_ch = out_ch
            
        self.features_net = nn.Sequential(*blocks)
        
        final_h = self.img_h // (2 ** len(channels_list))
        final_w = self.img_w // (2 ** len(channels_list))
        self.flatten_dim = final_h * final_w * channels_list[-1]
        
        self.flatten = nn.Flatten()
        self.head = nn.Sequential(
            nn.Linear(self.flatten_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        x = observations.float()
        if x.max() > 1.0:
            x = x / 255.0
            
        x = self.features_net(x)
        x = self.flatten(x)
        x = self.head(x)
        return x


# ============================================================
# 2. RESNET-18 (Torchvision Adaptado)
# ============================================================

class ResNet18Extractor(BaseFeaturesExtractor):
    """
    ResNet-18 Extractor adaptada para canais arbitrários e com cabeçalho de projeção linear.
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        self.model = models.resnet18(weights=None)
        
        n_frames = observation_space.shape[0]
        self.model.conv1 = nn.Conv2d(
            n_frames, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.model.fc = nn.Identity()
        
        self.head = nn.Sequential(
            nn.Linear(512, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        x = observations.float()
        if x.max() > 1.0:
            x = x / 255.0
        x = self.model(x)
        return self.head(x)
