import flwr as fl

def main():
        fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=50),
        strategy=fl.server.strategy.FedAvg(
            min_available_clients=3,
            min_fit_clients=3,
        )
    )

if __name__ == "__main__":
    main()
