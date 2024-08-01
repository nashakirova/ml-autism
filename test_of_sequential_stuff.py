import subprocess
for i in range(5,10):
    print(f"iteration {i}")
    subprocess.Popen('python server.py', shell=True)
    subprocess.Popen(f"python client.py {i}", shell=True)
    exit(0)