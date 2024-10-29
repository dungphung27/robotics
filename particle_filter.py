# particle_filter.py

import random
import math
import numpy as np
import pygame

class Particle:
    def __init__(self, x, y, theta, weight=1.0):
        self.x = x
        self.y = y
        self.theta = theta
        self.weight = weight

class ParticleFilter:
    def __init__(self, num_particles, map_surface, robot, sensor_range, sensor_fov):
        self.num_particles = num_particles
        self.map_surface = map_surface
        self.robot = robot
        self.sensor_range = sensor_range
        self.sensor_fov = sensor_fov
        self.particles = []
        self.init_particles()
        self.weights = np.ones(self.num_particles) / self.num_particles

    def init_particles(self):
        self.particles = []
        map_width, map_height = self.map_surface.get_size()
        while len(self.particles) < self.num_particles:
            x = random.uniform(0, map_width)
            y = random.uniform(0, map_height)
            theta = random.uniform(-math.pi, math.pi)
            # Check if the particle is in a free space
            if self.is_free_space(x, y):
                self.particles.append(Particle(x, y, theta))

    def is_free_space(self, x, y):
        # Check if the position is within the map boundaries and not on an obstacle
        if 0 <= int(x) < self.map_surface.get_width() and 0 <= int(y) < self.map_surface.get_height():
            color = self.map_surface.get_at((int(x), int(y)))
            return color == pygame.Color(255, 255, 255, 255)
        return False

    def predict(self, dt, control, motion_noise):
        for particle in self.particles:
            # Simulate motion with noise
            v_l = control['v_l'] + random.gauss(0, motion_noise['v'])
            v_r = control['v_r'] + random.gauss(0, motion_noise['v'])
            v = (v_r + v_l) / 2
            omega = -(v_r - v_l) / self.robot.L

            # Update state
            particle.x += v * math.sin(particle.theta) * dt
            particle.y -= v * math.cos(particle.theta) * dt  # Subtract due to inverted y-axis
            particle.theta += omega * dt

            # Normalize theta
            particle.theta = (particle.theta + math.pi) % (2 * math.pi) - math.pi

    def update(self, sensor_measurements, sensor_noise):
        weights = []
        for particle in self.particles:
            # Simulate sensor measurements from particle's position
            simulated_measurements = self.simulate_lidar(particle)
            # Compute weight based on similarity to actual measurements
            weight = self.compute_weight(sensor_measurements, simulated_measurements, sensor_noise)
            particle.weight = weight
            weights.append(weight)
        # Normalize weights
        weights = np.array(weights)
        sum_weights = np.sum(weights)
        if sum_weights != 0:
            weights /= sum_weights
        else:
            weights = np.ones(self.num_particles) / self.num_particles
        self.weights = weights

    def compute_weight(self, actual_measurements, simulated_measurements, sensor_noise):
        # Compare actual and simulated measurements
        weight = 1.0
        for a, s in zip(actual_measurements, simulated_measurements):
            if a['distance'] is not None and s['distance'] is not None:
                # Gaussian probability
                error = a['distance'] - s['distance']
                weight *= self.gaussian(0, sensor_noise, error)
            else:
                weight *= 0.1  # Low probability if no measurement
        return weight

    def gaussian(self, mu, sigma, x):
        # Gaussian probability density function
        return math.exp(- ((mu - x) ** 2) / (2 * sigma ** 2)) / (sigma * math.sqrt(2 * math.pi))

    def simulate_lidar(self, particle):
        # Simulate LIDAR measurements from the particle's perspective
        simulated_readings = []
        for angle_offset in self.robot.lidar.ray_angles:
            ray_angle = particle.theta + angle_offset
            sin_angle = math.sin(ray_angle)
            cos_angle = math.cos(ray_angle)

            distance = 0
            hit = False
            step_size = 1
            while distance < self.sensor_range:
                distance += step_size
                x = particle.x + distance * sin_angle
                y = particle.y - distance * cos_angle  # Subtract due to inverted y-axis

                if 0 <= int(x) < self.map_surface.get_width() and 0 <= int(y) < self.map_surface.get_height():
                    color = self.map_surface.get_at((int(x), int(y)))
                    if color != pygame.Color(255, 255, 255, 255):
                        hit = True
                        break
                else:
                    break

            if hit:
                simulated_readings.append({'distance': distance})
            else:
                simulated_readings.append({'distance': None})
        return simulated_readings

    def resample(self):
        # Systematic resampling
        cumulative_sum = np.cumsum(self.weights)
        step = 1.0 / self.num_particles
        start = random.uniform(0, step)
        positions = (start + np.arange(self.num_particles) * step) % 1.0

        indexes = np.searchsorted(cumulative_sum, positions)
        # Create new particles based on resampled indices
        new_particles = []
        for idx in indexes:
            particle = self.particles[idx]
            new_particle = Particle(particle.x, particle.y, particle.theta)
            new_particles.append(new_particle)
        self.particles = new_particles
        self.weights = np.ones(self.num_particles) / self.num_particles

    def estimate(self):
        # Estimate the state as the mean of the particles
        x = np.average([p.x for p in self.particles], weights=self.weights)
        y = np.average([p.y for p in self.particles], weights=self.weights)
        sin_thetas = np.average([math.sin(p.theta) for p in self.particles], weights=self.weights)
        cos_thetas = np.average([math.cos(p.theta) for p in self.particles], weights=self.weights)
        theta = math.atan2(sin_thetas, cos_thetas)
        return x, y, theta

    def draw(self, surface):
        # Draw particles
        for particle in self.particles:
            pygame.draw.circle(surface, (0, 255, 0), (int(particle.x), int(particle.y)), 2)
