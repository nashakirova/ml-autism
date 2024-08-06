import subprocess
import pandas as pd

dataset = pd.read_csv('./autism_data_client1.csv')
dataset.head()

if dataset.size > 5:
    print('TRAINING TO START')
    server = subprocess.Popen('python server.py', shell = True)
    client = subprocess.Popen('python ml-client.py', shell = True)
    server.wait()
    client.wait()
    exit(0)
exit(0)