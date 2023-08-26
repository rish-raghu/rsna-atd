import torchvision
import torch

def resnetForward(self, x, retFeatures=False):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.avgpool(x)
    features = torch.flatten(x, 1)
    x = self.fc(features)

    if retFeatures:
        return features, self.sigmoid(x)
    else:
        return self.sigmoid(x)

def getModel(args):
    if args.arch=='resnet50':
        model = torchvision.models.resnet50(num_classes=2)
        model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        return model
