import base64
import io
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from torch import nn


class Classifier_Prediction:
    """
    (1) Instantiate the model & tokenizer
    (2) Create_lenet
    (3) Inference
    (4) Predict & return list of prediction and probability

    """

    def __init__(self, input_image):
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
            print("No Cuda Available")

        self.save_path = "lenet.pth"
        self.path = input_image
        self.T = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

        self.create_lenet()
        self.inference()
        self.predict()

    def create_lenet(self):
        """ """
        self.model = nn.Sequential(
            nn.Conv2d(1, 6, 5, padding=2),
            nn.ReLU(),
            nn.AvgPool2d(2, stride=2),
            nn.Conv2d(6, 16, 5, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(2, stride=2),
            nn.Flatten(),
            nn.Linear(400, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        )
        lenet = self.model.to(self.device)
        lenet.load_state_dict(torch.load(self.save_path))
        self.lenet = lenet.eval()

    def inference(self):
        image_encoded = self.path.split(",")[1]
        image_bytes = io.BytesIO(base64.b64decode(image_encoded))
        img = Image.open(image_bytes).convert("L")
        img = img.resize((28, 28))
        x = (255 - np.expand_dims(np.array(img), -1)) / 255.0

        with torch.no_grad():
            pred = self.model(
                torch.unsqueeze(self.T(x), axis=0).float().to(self.device)
            )
        return F.softmax(pred, dim=-1).cpu().numpy()

    def predict(self):
        self.pred = self.inference()
        self.pred_idx = np.argmax(self.pred)
        self.prob = self.pred[0][self.pred_idx] * 100
        # self.prob = "{:.0%}".format(self.prob)

    def return_list(self):
        return self.pred_idx, self.prob


# poe force-cuda11
# pred_idx, prob = Classifier_Prediction("").return_list()
# print(pred_idx)
