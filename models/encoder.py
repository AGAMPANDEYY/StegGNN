import torch
import torch.nn as nn
from .layers.basic_layers import ImgToFeature, FeatureToImg
from .layers.attention import Block

class Encoder(nn.Module):
    def __init__(self, img_size: int, in_channels: int, embedding_dim: int) -> None:
        super(Encoder, self).__init__()

        # ─── FIX: save these onto self ─────────────────────────────────
        self.img_size = img_size
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        # ─────────────────────────────────────────────────────────────────

        self.img2feature = ImgToFeature(self.in_channels, self.embedding_dim)

        # Create five distinct Block instances, each with its own weights:
        self.blocks = nn.ModuleList([
            Block(
                embedding_dim    = self.embedding_dim,
                input_resolution = self.img_size // 8,
                dilation         = 1,
                stochastic       = False,
                epsilon          = 0.0
            )
            for _ in range(5)
        ])

        # Now it can use self.in_channels/self.embedding_dim here without error
        self.feature2img = FeatureToImg(self.in_channels, self.embedding_dim)

    def forward(self, secret: torch.Tensor, cover: torch.Tensor) -> torch.Tensor:
        features = self.img2feature(secret)  # → (B, embedding_dim, H/8, W/8)

        # Run each of the five Blocks in sequence (each has its own parameters):
        for blk in self.blocks:
            features = blk(features)

        secret_image_representation = self.feature2img(features)
        stego_image = secret_image_representation + cover
        return stego_image