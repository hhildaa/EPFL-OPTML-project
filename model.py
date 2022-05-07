import torch
import torchvision.models as models
import torch.nn as nn

def loadmodel():
    alexnet = models.alexnet(pretrained=True)
    alexnet.classifier[4] = nn.Linear(4096,1024)
    alexnet.classifier[6] = nn.Linear(1024,10)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    
    alexnet.device = device

    return alexnet.to(device)
