import copy
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from sklearn.metrics import confusion_matrix
from torch import nn, optim
from torchvision.transforms import ToTensor

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    print("No Cuda Available")
print(device)
