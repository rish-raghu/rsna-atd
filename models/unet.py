import torch
import torchvision
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


class UNet2D(nn.Module):
    def __init__(self, inChannels, depth=5):
        super(UNet2D, self).__init__()
        
        channels = 64
        self.downLayers = nn.ModuleList([DownConv2D(inChannels, 64)])
        for _ in range(depth-1):
            self.downLayers.append(DownConv2D(channels, 2*channels))
            channels *= 2
        
        self.upLayers = nn.ModuleList([])
        for _ in range(depth-1):
            self.upLayers.append(UpConv2D(channels, channels//2))
            channels //= 2
        
        self.linear = nn.Sequential(
            nn.Conv2d(channels, 1, 3),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.LazyLinear(64),
            nn.ReLU(),
        )

        self.bowelOut = nn.Linear(64, 2)
        self.extravOut = nn.Linear(64, 2)
        self.kidneyOut = nn.Linear(64, 3)
        self.liverOut = nn.Linear(64, 3)
        self.spleenOut = nn.Linear(64, 3)
        
    def forward(self, x):
        downMaps = []
        for layer in self.downLayers:
            x = layer(x)
            downMaps.append(x)
            x = nn.functional.max_pool2d(x, 2)
        
        x = downMaps[-1]
        for i, layer in enumerate(self.upLayers):
            x = layer(x, downMaps[len(downMaps)-i-2])
        x = self.linear(x)
        
        return self.bowelOut(x), self.extravOut(x), self.kidneyOut(x), self.liverOut(x), self.spleenOut(x)
        

class UNet3D(nn.Module):
    def __init__(self, depth=5):
        super(UNet3D, self).__init__()
        
        channels = 32
        self.downLayers = nn.ModuleList([DownConv3D(1, 32)])
        for _ in range(depth-1):
            self.downLayers.append(DownConv3D(channels, 2*channels))
            channels *= 2
        
        self.upLayers = nn.ModuleList([])
        for _ in range(depth-1):
            self.upLayers.append(UpConv3D(channels, channels//2))
            channels //= 2
        
        self.linear = nn.Sequential(
            #nn.Conv3d(channels, 1, 3),
            #nn.BatchNorm2d(1),
            #nn.ReLU(),
            nn.Flatten(),
            nn.LazyLinear(64),
            nn.ReLU(),
        )

        self.bowelOut = nn.Linear(64, 2)
        self.extravOut = nn.Linear(64, 2)
        self.kidneyOut = nn.Linear(64, 3)
        self.liverOut = nn.Linear(64, 3)
        self.spleenOut = nn.Linear(64, 3)
        
    def forward(self, x):
        downMaps = []
        for layer in self.downLayers:
            x = layer(x)
            downMaps.append(x)
            #x = nn.functional.max_pool3d(x, 2)
        
        x = downMaps[-1]
        for i, layer in enumerate(self.upLayers):
            x = layer(x, downMaps[len(downMaps)-i-2])
        x = self.linear(x)
        
        return self.bowelOut(x), self.extravOut(x), self.kidneyOut(x), self.liverOut(x), self.spleenOut(x)


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x = torch.ones((1, 1, 32, 128, 128)).to(device)
    network = UNet3D().to(device)
    for child in network.modules():
        if isinstance(child, nn.BatchNorm3d):
            print(child)
    assert False
    out = network(x)
