import torch
import torch.nn as nn
from torch.nn import Sequential as Seq, Conv2d

class BasicConv(Seq):
    def __init__(self, channels):
        m = []
        for i in range(1, len(channels)):
            m.append(Conv2d(channels[i - 1], channels[i], 1, bias=True, groups=4))
            m.append(nn.BatchNorm2d(channels[-1], affine=True))
            m.append(nn.PReLU(num_parameters=1, init=0.2))

        super(BasicConv, self).__init__(*m)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
      
    
class FFN(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(hidden_features),
        )
        self.activation = nn.PReLU(num_parameters=1, init=0.2)
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, in_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_features)
        )

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = x + shortcut
        return x
    

class ImgToFeature(nn.Module):
    """
    Takes an input image (C_in × H × W) and applies:
      Conv2d(in_channels → in_channels//4, stride=2)
      Conv2d(in_channels//4 → in_channels//2, stride=2)
      Conv2d(in_channels//2 → in_channels,   stride=2)
    Each conv is followed by BatchNorm2d and PReLU.
    """
    def __init__(self, in_channels: int, embedding_dim: int):
        super().__init__()
        # embedding_dim here is the “final” channel count
        mid1 = embedding_dim // 4
        mid2 = embedding_dim // 2

        self.net = nn.Sequential(
            # 1st layer: downsample ×2, channels → mid1
            nn.Conv2d(in_channels, mid1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(mid1),
            nn.PReLU(num_parameters=1, init=0.2),

            # 2nd layer: downsample ×2, channels → mid2
            nn.Conv2d(mid1, mid2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(mid2),
            nn.PReLU(num_parameters=1, init=0.2),

            # 3rd layer: downsample ×2, channels → embedding_dim
            nn.Conv2d(mid2, embedding_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embedding_dim),
            nn.PReLU(num_parameters=1, init=0.2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class FeatureToImg(nn.Module):
    """
    Takes a feature map (C_in × H × W) and upsamples it back to the original image size via:
      ConvTranspose2d(in_channels → in_channels//2, stride=2)
      ConvTranspose2d(in_channels//2 → in_channels//4, stride=2)
      ConvTranspose2d(in_channels//4 → out_channels,   stride=2)
    Each transpose‐conv is followed by BatchNorm2d + PReLU (except last, which uses Tanh).
    """
    def __init__(self, in_channels: int, embedding_dim: int):
        super().__init__()
        mid1 = embedding_dim // 2
        mid2 = embedding_dim // 4

        self.net = nn.Sequential(
            # 1st up‐conv: ×2 spatial, channels → mid1
            nn.ConvTranspose2d(embedding_dim, mid1, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(mid1),
            nn.PReLU(num_parameters=1, init=0.2),

            # 2nd up‐conv: ×2 spatial, channels → mid2
            nn.ConvTranspose2d(mid1, mid2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(mid2),
            nn.PReLU(num_parameters=1, init=0.2),

            # 3rd up‐conv: ×2 spatial, channels → in_channels (to reconstruct image)
            nn.ConvTranspose2d(mid2, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)