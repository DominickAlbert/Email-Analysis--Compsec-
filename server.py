import flwr as fl

# Start Flower server with  strategy=FedAvg (default)
if __name__ == "__main__":
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=3),
    )
