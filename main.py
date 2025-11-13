# main.py

import pygame
import config
from robot import Robot
from particle_filter import ParticleFilter
from utils import handle_keys, draw

def main():
    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((config.WIDTH, config.HEIGHT))
    pygame.display.set_caption("Particle Filter Localization with LiDAR")
    clock = pygame.time.Clock()
    running = True

    # Initialize robot and particle filter
    robot = Robot(config.WIDTH * 0.05, config.HEIGHT / 2, 0)
    pf = ParticleFilter(num_particles=config.num_particles, initial_pos=[robot.x, robot.y, robot.theta])

    while running:
        dt = clock.get_time() / 1000.0  # Delta time in seconds
        print(dt)
        if dt == 0:
            dt = 1e-16  # Avoid division by zero on the first frame

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Handle keyboard input
        v, omega = handle_keys()

        # Move robot
        robot.move(v, omega, dt)

        # Get robot's LiDAR measurements
        robot_measurements = robot.lidar_measurements(config.obstacles)

        # Particle filter steps
        pf.predict(v, omega, dt)
        pf.update(robot_measurements, config.obstacles)
        pf.resample()

        # Draw everything
        draw(screen, robot, pf.particles, config.obstacles, robot_measurements, pf)

        # Cap the frame rate
        clock.tick(60)
        

    pygame.quit()

if __name__ == "__main__":
    main()
