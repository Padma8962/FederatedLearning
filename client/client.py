from .preprocess import load_data
from .model import create_model
import flwr as fl
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score 
import numpy as np


class HospitalClient(fl.client.NumPyClient):
    def __init__(self, data_path):
        self.model = create_model()
        self.x_train, self.y_train = load_data(data_path)

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(
            self.x_train,
            self.y_train,
            epochs=20,
            batch_size=16,
            validation_split=0.2,
            verbose=1
        )
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        
        self.model.set_weights(parameters)

       
        y_pred = self.model.predict(self.x_train)
        y_pred = (y_pred > 0.5).astype(int).flatten()

        
        y_true = self.y_train

        
        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        
        loss, _ = self.model.evaluate(self.x_train, self.y_train, verbose=0)

        return float(loss), len(self.x_train), {
            "accuracy": float(acc),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1)
    }