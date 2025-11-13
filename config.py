# config.py

import numpy as np

# Screen dimensions
WIDTH, HEIGHT = 1280, 720
MAP_SIZE = 800          # px
SCALE = 50             # 1 mét = 100 px
CENTER = MAP_SIZE // 2  # tâm bản đồ
dt = 0.1                # thời gian mô phỏng (s)
#Color theme
background_color = (0,0,0) # black
robot_body_color = (255,255,255) # white
robot_head_color = (255, 0, 60)  # reddish
obstacles_color = (0, 0, 255)  # (246, 255, 0)

# LiDAR parameters
NUM_RAYS = 30        # LiDAR resolution: number of rays
LIDAR_RANGE = 50    # LiDAR range in pixels
LIDAR_NOISE_STD = 1  # Standard deviation of LiDAR noise

# Particle filter paramaters
num_particles = 200

# Motion noise parameters
MOTION_NOISE_LINEAR_STD = 0.5   # Linear motion noise standard deviation
MOTION_NOISE_ANGULAR_STD = 0.01  # Angular motion noise standard deviation

# Obstacles represented as Rect objects
# import pygame
# obstacles = [
#     pygame.Rect(200, 150, 100, 300),
#     pygame.Rect(500, 100, 50, 400),
#     pygame.Rect(300, 400, 200, 50)
# ]
