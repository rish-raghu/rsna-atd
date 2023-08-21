import torch
import torchvision
from torch import nn
import modules


def get_model(args):
    if args.arch=='unet2d':
        return UNet2D(args.z_size)
    elif args.arch=='unet3d':
        return UNet3D()


class UNet2D(nn.Module):
    def __init__(self, inChannels, depth=5):
        super(UNet2D, self).__init__()
        
        channels = 64
        self.downLayers = nn.ModuleList([modules.DownConv2D(inChannels, 64)])
        for _ in range(depth-1):
            self.downLayers.append(modules.DownConv2D(channels, 2*channels))
            channels *= 2
        
        self.upLayers = nn.ModuleList([])
        for _ in range(depth-1):
            self.upLayers.append(modules.UpConv2D(channels, channels//2))
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
        self.downLayers = nn.ModuleList([modules.DownConv3D(1, 32)])
        for _ in range(depth-1):
            self.downLayers.append(modules.DownConv3D(channels, 2*channels))
            channels *= 2
        
        self.upLayers = nn.ModuleList([])
        for _ in range(depth-1):
            self.upLayers.append(modules.UpConv3D(channels, channels//2))
            channels //= 2
        
        self.output = nn.Sequential(
            nn.Conv2d(channels, 1, 3),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d(1),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 5)
        )
        
    def forward(self, x):
        downMaps = []
        for layer in self.downLayers:
            x = layer(x)
            downMaps.append(x)
            x = nn.functional.max_pool3d(x, 2)
        
        x = downMaps[-1]
        for i, layer in enumerate(self.upLayers):
            x = layer(x, downMaps[len(downMaps)-i-2])
        
        return self.output(x)


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x = torch.ones((8, 32, 256, 256)).to(device)
    network = UNet2D(32).to(device)
    out = network(x)
    print(out)
