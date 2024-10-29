# robot.py

import math
import pygame
from sensors import LidarSensor

class DifferentialDriveRobot:
    def __init__(self, x, y, theta=0, L=40, v_max=200, a_max=200, image_path="robot_image.png", scale_factor=0.25):
        """
        Initialize the differential drive robot.
        """
        self.x = x
        self.y = y
        self.theta = theta  # Orientation angle in radians
        self.v_l = 0        # Left wheel velocity
        self.v_r = 0        # Right wheel velocity
        self.L = L          # Distance between the wheels (wheelbase)
        self.v_max = v_max  # Maximum wheel speed
        self.a_max = a_max  # Maximum wheel acceleration

        # Load and scale the robot image
        self.original_image = pygame.image.load(image_path).convert_alpha()
        self.original_image = pygame.transform.rotozoom(self.original_image, 0, scale_factor)

        # Set the initial image and rect
        self.image = self.original_image
        self.rect = self.image.get_rect(center=(self.x, self.y))

        # Initialize the LIDAR sensor with measurement error
        self.lidar = LidarSensor(self, measurement_error=2.0)

    def update(self, dt, a_l, a_r, map_surface):
        """
        Update the robot's state based on wheel accelerations.
        """
        # Update wheel velocities with acceleration limits
        self.v_l += a_l * dt
        self.v_r += a_r * dt

        # Limit wheel velocities to maximum speed
        self.v_l = max(-self.v_max, min(self.v_l, self.v_max))
        self.v_r = max(-self.v_max, min(self.v_r, self.v_max))

        # Compute robot's linear and angular velocities
        v = (self.v_r + self.v_l) / 2
        omega = -(self.v_r - self.v_l) / self.L

        # Update position and orientation
        self.x += v * math.sin(self.theta) * dt
        self.y -= v * math.cos(self.theta) * dt  # Subtract due to inverted y-axis
        self.theta += omega * dt

        # Normalize theta to keep it within -pi to pi
        self.theta = (self.theta + math.pi) % (2 * math.pi) - math.pi

        # Rotate the robot image to the new orientation
        angle_degrees = -math.degrees(self.theta)
        self.image = pygame.transform.rotozoom(self.original_image, angle_degrees, 1)
        self.rect = self.image.get_rect(center=(self.x, self.y))

        # Update the LIDAR sensor readings using the unmodified map_surface
        self.lidar.update(map_surface)

    def draw(self, surface):
        """
        Draw the robot on the given surface.
        """
        # Blit the rotated image onto the surface
        surface.blit(self.image, self.rect.topleft)
