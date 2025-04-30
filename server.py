import flwr as fl
from phe import paillier  # Import Paillier encryption
import numpy as np

public_key, private_key = paillier.generate_paillier_keypair()

class EncryptedFedAvg(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}
        
        # Extract encrypted parameters from clients
        encrypted_parameters = [res.parameters for _, res in results]
        
        # Decrypt and aggregate parameters
        aggregated_feature_log_prob = np.zeros_like(encrypted_parameters[0][0])
        aggregated_class_log_prior = np.zeros_like(encrypted_parameters[0][1])

        for params in encrypted_parameters:
            decrypted_feature_log_prob = np.array([private_key.decrypt(p) for p in params[0]])
            decrypted_class_log_prior = np.array([private_key.decrypt(p) for p in params[1]])
            aggregated_feature_log_prob += decrypted_feature_log_prob
            aggregated_class_log_prior += decrypted_class_log_prior

        # Average the parameters
        num_clients = len(encrypted_parameters)
        aggregated_feature_log_prob /= num_clients
        aggregated_class_log_prior /= num_clients

        return [aggregated_feature_log_prob, aggregated_class_log_prior], {}



# Start Flower server with  strategy=FedAvg (default)
if __name__ == "__main__":
    strategy = EncryptedFedAvg()
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )
