import flwr as fl
from client.model import create_model
import os

SAVE_DIR = "saved_models"

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
        server_address="127.0.0.1:8080",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=10),
    )