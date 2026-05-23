import flwr as fl
import os
from client.client import HospitalClient


if __name__ == "__main__":
    fl.client.start_numpy_client(
        server_address=os.environ.get("FLOWER_SERVER_ADDRESS", "127.0.0.1:8080"),
        client=HospitalClient("data/hospital3.csv"),
    )
