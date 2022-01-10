import torch
import torch.nn.functional as F
import torchvision
from torch import nn, optim
from torchvision.transforms import ToTensor

print(torch.cuda.is_available())
