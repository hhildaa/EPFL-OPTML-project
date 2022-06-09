import torch
import torchvision.models as models
import torch.nn as nn

def loadmodel():
    alexnet = models.alexnet(pretrained=True, dropout=0)
    alexnet.classifier[6] = nn.Linear(4096,10)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    
    alexnet.device = device

    return alexnet.to(device), device
