import flwr as fl
from client.client import HospitalClient


if __name__ == "__main__":
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=HospitalClient("data/hospital2.csv"),
    )