import sys
sys.path.insert(0,'/Users/nailyashakirova/opt/anaconda3/lib/python3.8/site-packages/')
import flwr as fl

def weighted_average(metrics):

   print(metrics)
   accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
   examples = [num_examples for num_examples, _ in metrics]
   return {"accuracy": int(sum(accuracies)) / int(sum(examples))}


strategy = fl.server.strategy.FedAvg(
   fraction_fit=1.0,  
   fraction_evaluate=1.0, 
   min_fit_clients=2,  
   min_evaluate_clients=2, 
   min_available_clients=2, 
   evaluate_metrics_aggregation_fn=weighted_average
)

fl.server.start_server(config=fl.server.ServerConfig(num_rounds=3), strategy=strategy)

# app = fl.server.ServerApp(
#     config=fl.server.ServerConfig(num_rounds=3),
#     strategy=strategy,
# )
