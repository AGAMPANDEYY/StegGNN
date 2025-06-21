import torch
import torch.nn as nn
from .graph_layers import WindowGrapher
from .basic_layers import FFN

### Squeeze and Excitation
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        short_cut = x
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)           # (B, C, 1, 1) → (B, C)
        y = self.fc(y).view(b, c, 1, 1)           # Apply MLP, reshape to (B, C, 1, 1)
        out = short_cut + (x * y)                 # residualconnection + Channel-wise scaling
        return out                        
    
class Block(nn.Module):
    """
    Encapsulates:
      • SEBlock(4*embedding_dim)
      • WindowGrapher(in_channels=embedding_dim, windows_size={4,8,16,32}, shift_size=0, input_resolution)
      • A 1×1 conv to go from (4*embedding_dim) back to embedding_dim
      • An FFN(in_features=embedding_dim, hidden_features=4*embedding_dim)

    Forward pass:
      1) run the same `x` through all four WindowGrapher’s
      2) concat along channel dim
      3) channel‐attention
      4) 1×1 conv
      5) FFN
    """
    def __init__(
        self,
        embedding_dim: int,
        input_resolution: int = 32,
        dilation: int = 1,
        stochastic: bool = False,
        epsilon: float = 0.0
    ):
        super().__init__()

        # 1) channel‐attention over (3 * embedding_dim)
        self.channel_attention = SEBlock(3 * embedding_dim)

        # 2) three WindowGrapher’s operating on `embedding_dim`‐channel feature maps
        self.graph_layer8  = WindowGrapher(
            in_channels=embedding_dim,
            windows_size=input_resolution//4,
            shift_size=input_resolution//8,
            input_resolution=(input_resolution,input_resolution),
            dilation=dilation,
            stochastic=stochastic,
            epsilon=epsilon,
        )
        self.graph_layer16 = WindowGrapher(
            in_channels=embedding_dim,
            windows_size=input_resolution//2,
            shift_size=input_resolution//4,
            input_resolution=(input_resolution,input_resolution),
            dilation=dilation,
            stochastic=stochastic,
            epsilon=epsilon,
        )
        self.graph_layer32 = WindowGrapher(
            in_channels=embedding_dim,
            windows_size=input_resolution,
            shift_size=0,
            input_resolution=(input_resolution,input_resolution),
            dilation=dilation,
            stochastic=stochastic,
            epsilon=epsilon,
        )

        # 3) 1×1 conv to reduce channels from (3*embedding_dim) → embedding_dim
        self.conv_intermediate = nn.Conv2d(
            in_channels= 3 * embedding_dim,
            out_channels=embedding_dim,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        # 4) FFN: (embedding_dim → 4*embedding_dim → embedding_dim)
        self.ffn = FFN(embedding_dim, hidden_features=4 * embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, embedding_dim, H, W)
        Returns: (B, embedding_dim, H, W) after the fusion block
        """
        # 1) apply each graph layer on the same input
        f8  = self.graph_layer8(x)  # → (B, embedding_dim, H, W)
        f16 = self.graph_layer16(x)
        f32 = self.graph_layer32(x)

        # 2) concatenate on channel dimension → (B, 4*embedding_dim, H, W)
        fused = torch.cat((f8, f16, f32), dim=1)

        # 3) channel‐attention
        fused = self.channel_attention(fused)

        # 4) 1×1 conv back to embedding_dim
        fused = self.conv_intermediate(fused)

        # 5) FFN
        fused = self.ffn(fused)

        return fused