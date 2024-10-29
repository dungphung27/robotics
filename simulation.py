# simulation.py

import pygame
from robot import DifferentialDriveRobot
from particle_filter import ParticleFilter

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
        pygame.display.set_caption("Differential Drive Robot Simulation with LIDAR and Particle Filter")

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

        # Create particle filter
        self.particle_filter = ParticleFilter(
            num_particles=20,
            map_surface=self.raw_map,
            robot=self.robot,
            sensor_range=self.robot.lidar.max_distance,
            sensor_fov=self.robot.lidar.fov
        )

        # Simulation loop control
        self.running = True

        # Control inputs
        self.a_l = 0
        self.a_r = 0

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

        self.a_l = a_l
        self.a_r = a_r

        # Update robot state using the unmodified map for sensing
        self.robot.update(self.dt, a_l, a_r, self.raw_map)

        # Prepare control inputs for particles
        control = {'v_l': self.robot.v_l, 'v_r': self.robot.v_r}

        # Predict particle states
        motion_noise = {'v': 5.0}  # Adjust noise level as needed
        self.particle_filter.predict(self.dt, control, motion_noise)

        # Get sensor measurements
        sensor_measurements = self.robot.lidar.get_measurements()

        # Update particle weights based on sensor measurements
        sensor_noise = 5.0  # Adjust sensor noise level as needed
        self.particle_filter.update(sensor_measurements, sensor_noise)

        # Resample particles
        self.particle_filter.resample()

        # Estimate the robot's position
        estimated_state = self.particle_filter.estimate()
        self.estimated_x, self.estimated_y, self.estimated_theta = estimated_state

    def render(self):
        """
        Render the simulation visuals.
        """
        # Clear the display map
        self.display_map.blit(self.raw_map, (0, 0))

        # Draw robot onto the display map
        self.robot.draw(self.display_map)

        # Draw particle filter
        self.particle_filter.draw(self.display_map)

        # Draw estimated position
        pygame.draw.circle(self.display_map, (0, 0, 255), (int(self.estimated_x), int(self.estimated_y)), 5)

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
