# https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment

import numpy as np
from scipy.special import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class Classifier:
    """
    (1) Instantiate the model & tokenizer
    (2) Preprocessing/encoding
    (3) Format scores
    (4) Return list of scores
    """

    def __init__(self, input_phrase):
        save_dir = "sentiment_analysis/nlp_classifier"
        model_name = "cardiffnlp/twitter-roberta-base-sentiment"
        self.model = AutoModelForSequenceClassification.from_pretrained(save_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.save_pretrained(save_dir)
        self.preprocessing(input_phrase)
        self.save_image()
        self.predict_output()

    def preprocessing(self, input_phrase):
        print("preprocessing")

    def save_image(self, input_phrase):
        print("save_image")

    def predict_output(self):
        self.prediciton = "x"

    def return_list(self):
        return [self.prediciton]
