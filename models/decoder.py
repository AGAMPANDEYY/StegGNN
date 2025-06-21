import torch
import torch.nn as nn
from .layers.basic_layers import ImgToFeature, FeatureToImg
from .layers.attention import Block

class Decoder(nn.Module):
    def __init__(self, img_size: int, in_channels: int, embedding_dim: int) -> None:
        super(Decoder, self).__init__()

        # ─── FIX: save these onto self ─────────────────────────────────
        self.img_size = img_size
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        # ─────────────────────────────────────────────────────────────────

        self.img2feature = ImgToFeature(self.in_channels, self.embedding_dim)

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

        self.feature2img = FeatureToImg(self.in_channels, self.embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.img2feature(x)
        for blk in self.blocks:
            features = blk(features)

        secret_image_reconstructed = self.feature2img(features)
        return secret_image_reconstructed