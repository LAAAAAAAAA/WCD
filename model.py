from torch import nn
from torchvision import models
from wtresnet50 import WTResNet50, Bottleneck 
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

def build_model(arch: str = "davit_tiny", pretrained: bool = True, num_classes: int = 2):
    """
    Build and return a model of the specified architecture.
    """
    if arch == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None).to(device)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif arch == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        model.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif arch =="wtresnet50":
        model = WTResNet50(Bottleneck, [3, 4, 6, 3], num_classes=2)
    
                
    return model
