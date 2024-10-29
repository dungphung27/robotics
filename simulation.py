# simulation.py

import pygame
from robot import DifferentialDriveRobot

class Simulation:
    def __init__(self, map_image_path='map.png', width=800, height=600, fps=60):
        """
        Initialize the simulation environment.
        """
        # Initialize Pygame
        pygame.init()

        # Screen dimensions
        self.WIDTH = width
        self.HEIGHT = height
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Differential Drive Robot Simulation with LIDAR")

        # Time parameters
        self.clock = pygame.time.Clock()
        self.fps = fps
        self.dt = 1 / self.fps

        # Load the map image
        self.raw_map = pygame.image.load(map_image_path).convert()
        self.raw_map = pygame.transform.scale(self.raw_map, (self.WIDTH, self.HEIGHT))

        # Create a display map to draw on (without modifying the raw map)
        self.display_map = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.display_map.blit(self.raw_map, (0, 0))

        # Create a surface for LIDAR visuals
        self.lidar_surface = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)

        # Create robot instance
        self.robot = DifferentialDriveRobot(x=self.WIDTH / 2, y=self.HEIGHT / 2)

        # Simulation loop control
        self.running = True

    def handle_events(self):
        """
        Handle user input and events.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

    def update(self):
        """
        Update the simulation state.
        """
        # Key states
        keys = pygame.key.get_pressed()

        # Wheel acceleration control
        a_l = 0
        a_r = 0
        a_max = self.robot.a_max

        # Linear acceleration input
        if keys[pygame.K_UP]:
            a_l += a_max
            a_r += a_max
        if keys[pygame.K_DOWN]:
            a_l -= a_max
            a_r -= a_max

        # Angular acceleration input
        if keys[pygame.K_LEFT]:
            a_l -= a_max
            a_r += a_max
        if keys[pygame.K_RIGHT]:
            a_l += a_max
            a_r -= a_max

        # Update robot state using the unmodified map for sensing
        self.robot.update(self.dt, a_l, a_r, self.raw_map)

    def render(self):
        """
        Render the simulation visuals.
        """
        # Clear the display map
        self.display_map.blit(self.raw_map, (0, 0))

        # Draw robot onto the display map
        self.robot.draw(self.display_map)

        # Clear the LIDAR surface
        self.lidar_surface.fill((0, 0, 0, 0))  # Transparent fill

        # Draw LIDAR visuals onto the LIDAR surface
        self.robot.lidar.draw(self.lidar_surface)

        # Blit the display map onto the main screen
        self.screen.blit(self.display_map, (0, 0))

        # Overlay the LIDAR surface onto the main screen
        self.screen.blit(self.lidar_surface, (0, 0))

        # Update display
        pygame.display.flip()

    def run(self):
        """
        Run the main simulation loop.
        """
        while self.running:
            self.handle_events()
            self.update()
            self.render()
            self.clock.tick(self.fps)

        # Quit Pygame
        pygame.quit()
