import torch
from torch import nn

class DownConv2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
    
    def forward(self, x):
        return self.layer(x)


class UpConv2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upconv = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels, in_channels//2, 2, padding='same'),
            nn.BatchNorm2d(in_channels//2),
            nn.ReLU(),
        )
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
    
    def forward(self, x, downMap):
        x = self.upconv(x)
        x = torch.cat((x, downMap), dim=1)
        return self.layer(x)


class DownConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
            nn.Conv3d(out_channels, out_channels, 3),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
        )
    
    def forward(self, x):
        return self.layer(x)


class UpConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upconv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear'),
            nn.Conv3d(in_channels, in_channels//2, 2),
            nn.BatchNorm3d(in_channels//2),
            nn.ReLU(),
        )
        self.layer = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
            nn.Conv3d(out_channels, out_channels, 3),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
        )
    
    def forward(self, x, downMap):
        x = self.upconv(x)
        x = torch.cat((x, downMap), dim=1)
        return self.layer(x)
