import subprocess
for i in range(5,10):
    print(f"iteration {i}")
    server = subprocess.Popen('python server.py', shell=True)
    client = subprocess.Popen(f"python client.py {i}", shell=True)
    server.wait()
    client.wait()
    print('HERE')
    #exit(0)