import sys
sys.path.insert(0,'/Users/nailyashakirova/opt/anaconda3/lib/python3.8/site-packages/')
import flwr as fl

fl.server.start_server(config=fl.server.ServerConfig(num_rounds=3))