# Model for SS304 dataset
#
# Based on ResNet50 for speed and accuracy
# replace the last layer with a fully connected layer
# with 6 output features

import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

def ss304_weld_model(out_features=6):
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.eval()
    
    for param in model.parameters():
        param.requires_grad = False   
        
    model.fc = nn.Sequential(
                nn.Linear(2048, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, out_features))
    
    return model