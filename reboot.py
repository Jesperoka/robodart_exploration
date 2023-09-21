# Convenience rebooting just incase the robot crashes while not connected to the desk
import panda_py

with open('sens.txt', 'r') as file:
    username = file.readline().strip()
    password = file.readline().strip()

panda_py.Desk("10.0.0.2", username, password).reboot()