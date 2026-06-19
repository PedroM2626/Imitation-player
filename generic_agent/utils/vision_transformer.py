"""
Vision Transformer (ViT) Feature Extractor para Imitation Learning.

Substitui a NatureCNN padrao do Stable-Baselines3 por um extrator baseado
em Self-Attention, capaz de capturar relacoes espaciais e temporais de forma
mais sofisticada.

Arquitetura:
    Input: (batch, 4, 128, 128) - 4 frames grayscale empilhados
    -> Patch Embedding (16x16 patches, 64 patches/frame, 256 tokens total)
    -> Positional Embedding + Temporal Embedding + CLS Token
    -> Transformer Encoder (4 layers, 4 heads, dim=256)
    -> CLS Token -> Linear -> features_dim (512)
"""

import torch as th
import torch.nn as nn
import numpy as np
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class PatchEmbedding(nn.Module):
    """Converte uma imagem em uma sequencia de patch embeddings."""
    
    def __init__(self, img_size: int = 128, patch_size: int = 16, 
                 in_channels: int = 1, embed_dim: int = 256):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2  # 64 patches para 128/16
        self.embed_dim = embed_dim
        
        # Projecao linear via Conv2d (equivalente a fatiar + linear)
        self.proj = nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=patch_size, stride=patch_size
        )
    
    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        Args:
            x: (batch, 1, H, W) - imagem grayscale de um unico frame
        Returns:
            (batch, num_patches, embed_dim) - sequencia de patch embeddings
        """
        # (B, 1, 128, 128) -> (B, 256, 8, 8)
        x = self.proj(x)
        # (B, 256, 8, 8) -> (B, 256, 64) -> (B, 64, 256)
        x = x.flatten(2).transpose(1, 2)
        return x


class TransformerEncoderBlock(nn.Module):
    """Um bloco do Transformer Encoder com Multi-Head Self-Attention e FFN."""
    
    def __init__(self, embed_dim: int = 256, num_heads: int = 4, 
                 mlp_ratio: float = 2.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        Args:
            x: (batch, seq_len, embed_dim)
        Returns:
            (batch, seq_len, embed_dim)
        """
        # Self-Attention com residual
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        # FFN com residual
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformerExtractor(BaseFeaturesExtractor):
    """
    Vision Transformer (ViT) Feature Extractor para Stable-Baselines3.
    
    Processa 4 frames grayscale empilhados usando self-attention para
    capturar relacoes espaciais (posicao do inimigo, obstaculos) e
    temporais (direcao do movimento, velocidade) simultaneamente.
    
    Compativel com o BC Trainer do imitation library.
    """
    
    def __init__(self, 
                 observation_space: gym.spaces.Box,
                 features_dim: int = 512,
                 embed_dim: int = 256,
                 patch_size: int = 16,
                 num_heads: int = 4,
                 num_layers: int = 4,
                 mlp_ratio: float = 2.0,
                 dropout: float = 0.1):
        super().__init__(observation_space, features_dim)
        
        # observation_space.shape = (n_frames, height, width) apos VecTransposeImage + VecFrameStack
        self.n_frames = observation_space.shape[0]      # 4
        self.img_height = observation_space.shape[1]     # 128
        self.img_width = observation_space.shape[2]      # 128
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        
        # Numero de patches por frame
        self.num_patches_per_frame = (self.img_height // patch_size) * (self.img_width // patch_size)
        # Total de patches (todos os frames)
        self.total_patches = self.n_frames * self.num_patches_per_frame  # 4 * 64 = 256
        # Total de tokens (patches + CLS)
        self.total_tokens = self.total_patches + 1  # 257
        
        # Patch embedding (compartilhado entre frames)
        self.patch_embed = PatchEmbedding(
            img_size=self.img_height,
            patch_size=patch_size,
            in_channels=1,
            embed_dim=embed_dim,
        )
        
        # CLS token (token especial que resume toda a cena)
        self.cls_token = nn.Parameter(th.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Positional embedding (posicao espacial de cada patch)
        self.pos_embed = nn.Parameter(th.zeros(1, self.total_tokens, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Temporal embedding (qual frame cada patch pertence: 0, 1, 2 ou 3)
        self.temporal_embed = nn.Parameter(th.zeros(1, self.n_frames, embed_dim))
        nn.init.trunc_normal_(self.temporal_embed, std=0.02)
        
        # Dropout apos embeddings
        self.embed_dropout = nn.Dropout(dropout)
        
        # Transformer Encoder
        self.encoder = nn.Sequential(*[
            TransformerEncoderBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        
        # Layer norm final
        self.norm = nn.LayerNorm(embed_dim)
        
        # Projecao para features_dim
        self.head = nn.Sequential(
            nn.Linear(embed_dim, features_dim),
            nn.ReLU(),
        )
    
    def forward(self, observations: th.Tensor) -> th.Tensor:
        """
        Args:
            observations: (batch, n_frames, H, W) - frames empilhados
        Returns:
            (batch, features_dim) - vetor de features para o policy head
        """
        batch_size = observations.shape[0]
        device = observations.device
        
        # Normalizar pixel values para [0, 1]
        x = observations.float()
        if x.max() > 1.0:
            x = x / 255.0
        
        # Processar cada frame individualmente pelo patch embedding
        all_patches = []
        for f in range(self.n_frames):
            # (B, H, W) -> (B, 1, H, W) para o Conv2d
            frame = x[:, f:f+1, :, :]
            # (B, 1, H, W) -> (B, num_patches, embed_dim)
            patches = self.patch_embed(frame)
            # Adicionar temporal embedding para este frame
            patches = patches + self.temporal_embed[:, f:f+1, :]
            all_patches.append(patches)
        
        # Concatenar todos os patches: (B, total_patches, embed_dim)
        x = th.cat(all_patches, dim=1)
        
        # Adicionar CLS token no inicio
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = th.cat([cls_tokens, x], dim=1)  # (B, total_tokens, embed_dim)
        
        # Adicionar positional embedding
        x = x + self.pos_embed
        x = self.embed_dropout(x)
        
        # Passar pelo Transformer Encoder
        x = self.encoder(x)
        
        # Extrair o CLS token (primeiro token)
        x = self.norm(x[:, 0])
        
        # Projetar para features_dim
        features = self.head(x)
        
        return features
