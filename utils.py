# utils.py

import pygame
import numpy as np
from config import NUM_RAYS
from robot import Robot
from particle_filter import ParticleFilter
import math
# utils.py
import config

import pygame
import numpy as np


# Colors
BLACK = (0, 0, 0)
LIGHT_GREEN = (173, 255, 47)
WHITE = (255, 255, 255)


def handle_keys():
    keys = pygame.key.get_pressed()
    v = 0      # Linear velocity
    omega = 0  # Angular velocity

    if keys[pygame.K_UP]:
        v = 100  # Move forward
    if keys[pygame.K_DOWN]:
        v = -100  # Move backward
    if keys[pygame.K_LEFT]:
        omega = -np.pi  # Rotate left
    if keys[pygame.K_RIGHT]:
        omega = np.pi  # Rotate right

    return v, omega

def draw(screen, robot, particles, obstacles, robot_measurements, pf):
    screen.fill(config.background_color)  # Clear screen with white

    # Draw obstacles
    for obstacle in obstacles:
        pygame.draw.rect(screen, config.obstacles_color, obstacle)

    # Optionally draw LiDAR rays
    angles = np.linspace(0, 2 * np.pi, NUM_RAYS, endpoint=False)
    for i, distance in enumerate(robot_measurements):
        angle = robot.theta + angles[i]
        x_end = robot.x + distance * np.cos(angle)
        y_end = robot.y + distance * np.sin(angle)
        # draw lidar lines
        pygame.draw.line(screen, (200, 200, 200), (robot.x, robot.y), (x_end, y_end), 1)
        # Draw a red dot at the end of each LiDAR ray
        if distance < config.LIDAR_RANGE*0.8:
            pygame.draw.circle(screen, (255, 0, 0), (int(x_end), int(y_end)), 1)

    # Draw particles
    for particle in particles:
        pygame.draw.circle(screen, (0, 255, 0), (int(particle[0]), int(particle[1])), 2)

    # Draw robot's path
    if len(robot.path) > 1:
        pygame.draw.lines(screen, (255, 0, 0), False, robot.path, 2)

    # Draw estimated path
    if len(pf.estimated_path) > 1:
        pygame.draw.lines(screen, (128, 0, 128), False, pf.estimated_path, 2)

    # Draw robot
    draw_triangle_with_t(screen,(int(robot.x), int(robot.y)),scale=0.2,angle=math.degrees(robot.theta))
    # Draw robot orientation
    end_x = robot.x + 15 * np.cos(robot.theta)
    end_y = robot.y + 15 * np.sin(robot.theta)
    #pygame.draw.line(screen, (0, 0, 255), (robot.x, robot.y), (end_x, end_y), 2)

    # Draw estimated position
    est_x, est_y = pf.estimated_position
    #pygame.draw.circle(screen, (128, 0, 128), (int(est_x), int(est_y)), 5)

    pygame.display.flip()


def draw_triangle_with_t(surface, pos, scale=1.0, angle=0, colors=None):
    """
    Draws a scalable and rotatable triangular robot pointing upwards with an orientation indicator.

    Args:
        surface (pygame.Surface): The surface to draw on.
        pos (tuple): (cx, cy) position of the center.
        scale (float): Scale factor for the size of the robot.
        angle (float): Rotation angle in degrees (counter-clockwise).
        colors (dict): A dictionary of colors with keys 'body' and 'indicator' (default colors used if None).
    """
    import math
    import pygame

    cx, cy = pos
    # Default colors
    if colors is None:
        colors = {
            "body": (255, 255, 255),       # Light Green
            "indicator": (255, 69, 0)     # Orange Red
        }

    # Triangle settings
    size = 70 * scale  # Length from center to vertex

    # Define the triangle vertices before rotation
    # Adjusted angles so the triangle points up when angle=0
    vertex_angles = [0, 120, 240]

    # Convert rotation angle to radians
    rotation_rad = math.radians(angle)

    # Calculate the vertices after rotation
    vertices = []
    for vertex_angle in vertex_angles:
        total_angle = math.radians(vertex_angle) + rotation_rad
        x = cx + size * math.cos(total_angle)
        y = cy + size * math.sin(total_angle)
        vertices.append((x, y))

    # Draw the robot body (triangle)
    pygame.draw.polygon(surface, colors["body"], vertices)

    # Calculate midpoints of sides adjacent to the front vertex
    # Front vertex is vertices[0]
    mid1_x = (vertices[0][0] + vertices[1][0]) / 2
    mid1_y = (vertices[0][1] + vertices[1][1]) / 2

    mid2_x = (vertices[0][0] + vertices[2][0]) / 2
    mid2_y = (vertices[0][1] + vertices[2][1]) / 2

    # Define the indicator polygon
    indicator_vertices = [
        (vertices[0][0], vertices[0][1]),
        (mid1_x, mid1_y),
        (mid2_x, mid2_y)
    ]

    # Draw the orientation indicator
    pygame.draw.polygon(surface, colors["indicator"], indicator_vertices)

