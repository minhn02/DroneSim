import numpy as np
import serial
import time
import arty_utils

class ArtyAgent:
    def __init__(self, port='/dev/ttyUSB4', baudrate=9600):
        self.port = port
        self.baudrate = baudrate
        self.connection = None
        self.connect()

    def send_observation(self, observation, timestep):
        state_str = f"{observation[0]} {observation[1]} {observation[2]} {observation[3]} {observation[4]} {observation[5]} {observation[6]} {observation[7]} {observation[8]} {observation[9]} {observation[10]} {observation[11]} {timestep}\n"
        print("sending: ", state_str)
        self.connection.write(state_str.encode())

    def read_action(self):
        action_str = self.connection.readline().decode().strip()

        # Split the content by commas to get individual float strings
        int_strings = action_str.split(' ')
        float_inputs = []
        
        for i in range(4):
            int_value = int(int_strings[i])
            float_inputs.append(arty_utils.reinterpret_int_as_float(int_value))

        return float_inputs

    def act(self, state, timestep):
        self.send_observation(state, timestep)
        action = self.read_action()
        return action

    def connect(self):
        self.connection = serial.Serial(self.port, self.baudrate)
        # Wait for the connection to establish
        time.sleep(2)