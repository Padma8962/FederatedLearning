import flwr as fl
from client.preprocess import load_data
from client.model import create_model


class HospitalClient(fl.client.NumPyClient):
    def __init__(self, data_path):
        self.model = create_model()
        self.x_train, self.y_train = load_data(data_path)

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(self.x_train, self.y_train, epochs=3, batch_size=16, verbose=0)
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_train, self.y_train, verbose=0)
        return float(loss), len(self.x_train), {"accuracy": float(accuracy)}