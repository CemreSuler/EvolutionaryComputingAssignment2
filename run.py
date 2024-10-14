import subprocess

# NOTE CHANGE COMMAND TO PYTHON VERSION
for i in range(10):
    subprocess.Popen(["python3.11", "algorithm_1.py"], start_new_session=True)

for i in range(10):
    subprocess.Popen(["python3.11", "algorithm_2.py"], start_new_session=True)
