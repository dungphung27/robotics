# sensors.py

import math
import pygame
import numpy as np

class LidarSensor:
    def __init__(self, robot, num_rays=70, max_distance=200, fov=360, point_life=255, measurement_error=1.0):
        """
        Initialize the LIDAR sensor.
        """
        self.robot = robot
        self.num_rays = num_rays
        self.max_distance = max_distance
        self.fov = math.radians(fov)
        self.point_life = point_life
        self.measurement_error = measurement_error
        self.ray_angles = []
        self.readings = []  # Each reading is a dict with 'position' and 'error'
        self.hit_points = []  # List to store hit points and their ages
        self._calculate_ray_angles()

    def _calculate_ray_angles(self):
        """
        Calculate the angles at which rays will be cast.
        """
        start_angle = -self.fov / 2
        angle_increment = self.fov / self.num_rays
        self.ray_angles = [start_angle + i * angle_increment for i in range(self.num_rays)]

    def update(self, map_surface):
        """
        Update the sensor readings by casting rays and detecting intersections.
        """
        self.readings = []
        for angle in self.ray_angles:
            ray_angle = self.robot.theta + angle
            sin_angle = math.sin(ray_angle)
            cos_angle = math.cos(ray_angle)

            # Initialize variables for ray casting
            step_size = 1  # Incremental step for ray casting
            distance = 0
            hit = False

            # Cast the ray incrementally
            while distance < self.max_distance:
                distance += step_size
                x = self.robot.x + distance * sin_angle
                y = self.robot.y - distance * cos_angle  # Subtract due to inverted y-axis

                # Check if the point is within the map boundaries
                if 0 <= int(x) < map_surface.get_width() and 0 <= int(y) < map_surface.get_height():
                    color = map_surface.get_at((int(x), int(y)))
                    if color != pygame.Color(255, 255, 255, 255):
                        hit = True
                        break
                else:
                    # Ray is outside the map boundaries
                    break

            if hit:
                # Introduce measurement error
                measured_distance = distance + np.random.normal(0, self.measurement_error)
                measured_distance = max(0, min(measured_distance, self.max_distance))

                # Calculate the point with error
                x = self.robot.x + measured_distance * sin_angle
                y = self.robot.y - measured_distance * cos_angle

                # Store the reading with error
                self.readings.append({'position': (x, y), 'error': measured_distance - distance, 'distance': measured_distance})

                # Add the hit point with an initial age of 0
                self.hit_points.append({'position': (x, y), 'age': 0})
            else:
                # No hit within max_distance
                x = self.robot.x + self.max_distance * sin_angle
                y = self.robot.y - self.max_distance * cos_angle
                self.readings.append({'position': (x, y), 'error': None, 'distance': None})

        # Update the ages of the hit points
        for point in self.hit_points:
            point['age'] += 1

        # Remove points that have exceeded their life
        self.hit_points = [point for point in self.hit_points if point['age'] <= self.point_life]

    def draw(self, surface):
        """
        Visualize the LIDAR sensor rays and hit points on the given surface.
        """
        SENSOR_COLOR = (255, 0, 0)  # Red color for rays

        # Draw LIDAR rays
        for reading in self.readings:
            point = reading['position']
            pygame.draw.line(surface, SENSOR_COLOR, (int(self.robot.x), int(self.robot.y)), (int(point[0]), int(point[1])), 1)

        # Draw hit points with fading effect
        for point in self.hit_points:
            age_ratio = point['age'] / self.point_life
            alpha_value = max(0, int(255 * (1 - age_ratio)))  # Fade from fully opaque to transparent
            point_color = (255, 0, 0, alpha_value)  # RGBA with fading alpha

            # Create a small surface for the point with per-pixel alpha
            point_surface = pygame.Surface((4, 4), pygame.SRCALPHA)
            pygame.draw.circle(point_surface, point_color, (2, 2), 2)
            surface.blit(point_surface, (int(point['position'][0]) - 2, int(point['position'][1]) - 2))

    def get_measurements(self):
        """
        Return the sensor measurements needed for the particle filter.
        """
        measurements = []
        for reading in self.readings:
            measurements.append({'distance': reading['distance']})
        return measurements
