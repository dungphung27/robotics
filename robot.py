# robot.py

import numpy as np
from config import NUM_RAYS, LIDAR_RANGE, LIDAR_NOISE_STD, MOTION_NOISE_LINEAR_STD, MOTION_NOISE_ANGULAR_STD

class Robot:
    def __init__(self, x = 0, y = 0, theta =0):
        self.x = x        # X position
        self.y = y        # Y position
        self.theta = theta  # Orientation
        self.path = [(x, y)]  # Store robot's path
    def move(self, v, omega, dt):
        # Motion noise standard deviations
       

        # Add motion noise to control inputs
        v_noisy = v 
        omega_noisy = omega 

        # State vector: [x, y, theta]^T
        state = np.array([self.x, self.y, self.theta])

        # Control input vector with noise: [v_noisy, omega_noisy]^T
        control_noisy = np.array([v_noisy, omega_noisy])

        # Control input matrix B(theta)
        B = np.array([
            [np.cos(self.theta), 0],
            [np.sin(self.theta), 0],
            [0, 1]
        ])

        # Update state
        state = state + (B @ control_noisy) * dt

        # Update the object's state variables
        self.x, self.y, self.theta = state
        self.path.append((self.x, self.y))  # Append current position to path
    def robot_measurements(self,lidar):
        measurements = []
        for i in lidar:
            dist = i * 50
            measurements.append(dist)
        return np.array(measurements)