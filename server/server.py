import flwr as fl
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from client.model import create_model

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SAVE_DIR = os.path.join(PROJECT_DIR, "saved_models")
SERVER_HOST = os.environ.get("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.environ.get("SERVER_PORT", os.environ.get("PORT", 8080)))

os.makedirs(SAVE_DIR, exist_ok=True)


class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None:
            print(f"\nSaving global model at round {server_round}")

            model = create_model()

            weights = fl.common.parameters_to_ndarrays(aggregated_parameters)
            model.set_weights(weights)

            model_path = os.path.join(
                SAVE_DIR, f"model_round_{server_round}.keras"
            )
            model.save(model_path)
            print(f"Global model saved at: {model_path}")

            if metrics:
                print(f"Round {server_round} metrics: {metrics}")
            else:
                print(f"Round {server_round}: No metrics returned")

        return aggregated_parameters, metrics


if __name__ == "__main__":
    strategy = SaveModelStrategy()

    fl.server.start_server(
        server_address=f"{SERVER_HOST}:{SERVER_PORT}",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=10),
    )
