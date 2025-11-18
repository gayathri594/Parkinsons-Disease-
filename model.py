import torch.nn as nn
from torchvision import models

def get_model():
    model = models.vgg19(pretrained=True)
    for param in model.features.parameters():
        param.requires_grad = False
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, 2)
    return model
