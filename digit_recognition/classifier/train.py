# https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment

import copy
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from scipy.special import softmax
from torch import nn, optim
from torchvision.transforms import ToTensor
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class Train_Classifier:
    """
    (1) Instantiate the model & tokenizer
    (2) Preprocessing/encoding
    (3) Format scores
    (4) Return list of scores

    """

    def __init__(self, input_phrase):
        """
        (1) Instantiate the model & tokenizer

        """

        self.T = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        save_path = "lenet.pth"

        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
            print("No Cuda Available")
        lenet = train(40, device=device)
        torch.save(lenet.state_dict(), "lenet.pth")

        self.dataloader()
        self.model = self.create_lenet().to(device)

        self.preprocessing(input_phrase)

        self.predict_output()

    def dataloader(self):
        """
        (1) Instantiate the model & tokenizer

        """
        # Get data from T.V
        numb_batch = 64
        self.train_data = torchvision.datasets.MNIST(
            "mnist_data",
            train=True,
            download=True,
            transform=self.T,
        )
        self.val_data = torchvision.datasets.MNIST(
            "mnist_data",
            train=False,
            download=True,
            transform=self.T,
        )

        self.train_dl = torch.utils.data.DataLoader(
            self.train_data,
            batch_size=numb_batch,
        )
        self.val_dl = torch.utils.data.DataLoader(
            self.val_data,
            batch_size=numb_batch,
        )

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

    def validate(model, data):
        total = 0
        correct = 0
        for i, (images, labels) in enumerate(data):
            images = images.cuda()
            x = model(images)
            value, pred = torch.max(x, 1)
            pred = pred.data.cpu()
            total += x.size(0)
            correct += torch.sum(pred == labels)
        return correct * 100.0 / total

    def train(numb_epoch=3, lr=1e-3, device="cpu"):
        accuracies = []
        cnn = create_lenet().to(device)
        cec = nn.CrossEntropyLoss()
        optimizer = optim.Adam(cnn.parameters(), lr=lr)
        max_accuracy = 0
        for epoch in range(numb_epoch):
            for i, (images, labels) in enumerate(train_dl):
                images = images.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                pred = cnn(images)
                loss = cec(pred, labels)
                loss.backward()
                optimizer.step()
            accuracy = float(validate(cnn, val_dl))
            accuracies.append(accuracy)
            if accuracy > max_accuracy:
                best_model = copy.deepcopy(cnn)
                max_accuracy = accuracy
                print("Saving Best Model with Accuracy: ", accuracy)
            print("Epoch:", epoch + 1, "Accuracy :", accuracy, "%")
        plt.plot(accuracies)
        return best_model

    def inference(path, model, device):
        r = requests.get(path)
        with BytesIO(r.content) as f:
            img = Image.open(f).convert(mode="L")
            img = img.resize((28, 28))
            x = (255 - np.expand_dims(np.array(img), -1)) / 255.0
        with torch.no_grad():
            pred = model(torch.unsqueeze(T(x), axis=0).float().to(device))
        return F.softmax(pred, dim=-1).cpu().numpy()

    def predict_dl(model, data):
        y_pred = []
        y_true = []
        for i, (images, labels) in enumerate(data):
            images = images.cuda()
            x = model(images)
            value, pred = torch.max(x, 1)
            pred = pred.data.cpu()
            y_pred.extend(list(pred.numpy()))
            y_true.extend(list(labels.numpy()))
        return np.array(y_pred), np.array(y_true)

    def preprocessing(self, input_phrase):
        print("preprocessing")

    def save_image(self, input_phrase):
        print("save_image")

    def predict_output(self):
        self.prediciton = "x"

    def return_list(self):
        return [self.prediciton]

        lenet.load_state_dict(torch.load(save_path))
        lenet.eval()
