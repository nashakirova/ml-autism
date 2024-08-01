
import flwr as fl
from typing import Optional, List
import numpy as np

def weighted_average(metrics):

   print(metrics)
   accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
   examples = [num_examples for num_examples, _ in metrics]
   return {"accuracy": int(sum(accuracies)) / int(sum(examples))}

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        rnd: int,
        results,
        failures,
    ) -> Optional[fl.common.Parameters]:
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(rnd, results, failures)
        if aggregated_parameters is not None:
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)
            np.savez(f"weights.npz", *aggregated_ndarrays)
        return aggregated_parameters, aggregated_metrics
    def initialize_parameters(self, client_manager) -> fl.common.Parameters:
        state_dict = np.load("weights.npz")
        parameters = fl.common.ndarrays_to_parameters(state_dict.values())
        return parameters

strategy = SaveModelStrategy(
   fraction_fit=1.0,  
   fraction_evaluate=1.0, 
   min_fit_clients=2,  
   min_evaluate_clients=1, 
   min_available_clients=1, 
   evaluate_metrics_aggregation_fn=weighted_average
)

fl.server.start_server(config=fl.server.ServerConfig(num_rounds=3), strategy=strategy)

# app = fl.server.ServerApp(
#     config=fl.server.ServerConfig(num_rounds=3),
#     strategy=strategy,
# )
