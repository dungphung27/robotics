import numpy as np
from config import NUM_RAYS, LIDAR_RANGE, LIDAR_NOISE_STD
from joblib import Parallel, delayed
import cv2
from numba import njit

class ParticleFilter:
    def __init__(self, num_particles,map_img,initial_pos):
        self.num_particles = num_particles
        self.particles = np.empty((num_particles, 3))  # Each particle has x, y, theta
        self.weights = np.ones(self.num_particles) / self.num_particles
        self.initialize_particles(map_img)
        self.estimated_path = []  # Store estimated positions
        self.estimated_position = (0,0)

    def initialize_particles(self, map_img):
    # --- Chuyển sang grayscale để dễ kiểm tra vùng trống ---
        if len(map_img.shape) == 3:
            gray = cv2.cvtColor(map_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = map_img.copy()

        map_height, map_width = gray.shape[:2]
        count = 0
        max_tries = self.num_particles * 10  

        while count < self.num_particles and max_tries > 0:
            max_tries -= 1
            px = np.random.uniform(0, map_width)
            py = np.random.uniform(0, map_height)

            # --- Kiểm tra vùng trắng (vùng trống) ---
            if gray[int(py), int(px)] > 250:
                theta = 0
                self.particles[count] = [px, py, theta]
                count += 1

        if count < self.num_particles:
            print(f"[Cảnh báo] Chỉ tạo được {count}/{self.num_particles} hạt hợp lệ!")

    def predict(self, v, omega, dt):
        self.particles[:, 2] += omega * dt
        self.particles[:, 0] += v * np.cos(self.particles[:, 2]) * dt * 50      
        self.particles[:, 1] += v * np.sin(self.particles[:, 2]) * dt * 50

    def simulate_lidar(self, particles, img):
        num_particles = particles.shape[0]
        angles = np.linspace(0, 2 * np.pi, NUM_RAYS, endpoint=False)

        # 🔹 Chuyển ảnh sang grayscale 1 lần duy nhất (không làm trong mỗi vòng lặp)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        def process_particle(i):
            x, y, theta = particles[i]
            total_angles = theta + angles
            distances = np.array([
                self.cast_ray(x, y, total_angle, img_gray)  # ✅ truyền ảnh xám
                for total_angle in total_angles
            ])
            return distances

        results = Parallel(n_jobs=7)(
            delayed(process_particle)(i) for i in range(num_particles)
        )
        measurements = np.stack(results)
        return measurements
    def cast_ray(self, x0, y0, angle, img_gray):
        LIDAR_RANGE_MIN = 0.11  # mét
        LIDAR_RANGE_MAX = 3.5   # mét
        SCALE = 50             # 1 mét = 100 px
        step = 10               # bước quét (pixel)
        max_dist_px = int(LIDAR_RANGE_MAX * SCALE)

        h, w = img_gray.shape[:2]

        for dist_px in range(0, max_dist_px, step):
            x = int(x0 + np.cos(angle) * dist_px)
            y = int(y0 + np.sin(angle) * dist_px)

            # --- Nếu vượt biên, coi như gặp vật cản ---
            if x < 0 or y < 0 or x >= w or y >= h:
                dist_m = dist_px / SCALE
                return dist_m if LIDAR_RANGE_MIN <= dist_m <= LIDAR_RANGE_MAX else 0.0

            # --- Nếu gặp vật cản (pixel tối) ---
            if img_gray[y, x] < 10:
                dist_m = dist_px / SCALE
                return dist_m if LIDAR_RANGE_MIN <= dist_m <= LIDAR_RANGE_MAX else 0.0

        # --- Không gặp vật cản trong tầm quét ---
        return 0.0
    def update(self, robot_measurements,img):
        # print("a")
        particle_measurements = self.simulate_lidar(self.particles,img) * 50  # Chuyển sang cm
        # Góc thực tế của mỗi phép đo
        indices = np.linspace(0, len(robot_measurements) - 1, particle_measurements.shape[1]).astype(int)
        robot_sampled = robot_measurements[indices]

        mse = np.mean((particle_measurements - robot_sampled) ** 2, axis=1)
        self.weights = np.exp(-mse / (200 * LIDAR_NOISE_STD ** 2))
        self.weights += 1e-300  # Avoid zeros
        self.weights /= sum(self.weights)

    def resample(self):
        # --- Systematic Resampling ---
        N = self.num_particles
        positions = (np.arange(N) + np.random.rand()) / N
        cumulative_sum = np.cumsum(self.weights)
        indices = np.zeros(N, dtype=int)

        i, j = 0, 0
        while i < N:
            if positions[i] < cumulative_sum[j]:
                indices[i] = j
                i += 1
            else:
                j += 1
        # print(indices)
        self.particles = self.particles[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles

        # Estimate position (weighted mean trước khi reset)
        x_estimate = np.mean(self.particles[:, 0])
        y_estimate = np.mean(self.particles[:, 1])
        self.estimated_position = (x_estimate, y_estimate)
        self.estimated_path.append(self.estimated_position)
