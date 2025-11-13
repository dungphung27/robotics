"""
particle_filter_map.py

Chạy: python particle_filter_map.py
Phụ thuộc: numpy, opencv-python

Hướng dẫn:
- Mở cửa sổ map, click chuột trái để đặt vị trí robot (pixel).
- Nhấn 'l' để (re)khởi tạo particles (ngẫu nhiên trên vùng free).
- Nhấn 's' để chạy 1 bước particle filter.
- Nhấn 'a' để chạy tự động (toggle).
- Nhấn 'r' để xoá robot (bỏ chọn).
- Nhấn 'q' để thoát.
"""
import cv2
import numpy as np
import math
import time
import sys

# ======== CÀI ĐẶT ========
MAP_FILENAME = "map.png"   # thay bằng file map của bạn (png)
WINDOW_NAME = "Particle Filter Map"
NUM_PARTICLES = 800
RAY_COUNT = 24             # số tia measurement dùng để so sánh
MAX_RANGE_PX = 300         # khoảng cách tối đa dò (pixel)
SENSOR_NOISE = 6.0         # sigma (pixel) mô phỏng nhiễu cảm biến
MOTION_NOISE_POS = 1.5     # sigma position khi predict (pixel)
MOTION_NOISE_ANGLE = 0.05  # sigma heading khi predict (rad)

# ======== TẢI MAP VÀ TIỀN XỬ LÝ ========
orig = cv2.imread(MAP_FILENAME, cv2.IMREAD_COLOR)
if orig is None:
    print(f"Không tìm thấy file map: {MAP_FILENAME}")
    sys.exit(1)

# Chuyển sang grayscale để phát hiện obstacles (pixel tối = obstacle)
gray_map = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
# Ngưỡng: pixel < 128 là obstacle
occ = (gray_map < 128).astype(np.uint8)  # 1 = obstacle, 0 = free

H, W = gray_map.shape
center = (W // 2, H // 2)

# ======== Hàm tiện ích ========
def is_free(px, py):
    if px < 0 or px >= W or py < 0 or py >= H: 
        return False
    return occ[py, px] == 0

def random_free_point():
    # chọn ngẫu nhiên trên vùng free
    while True:
        x = np.random.randint(0, W)
        y = np.random.randint(0, H)
        if is_free(x, y):
            return x, y

def ray_distance_px(x, y, angle, max_range=MAX_RANGE_PX):
    """
    Tracing ray từ (x,y) theo góc angle (radian) trên grid occ.
    Trả về khoảng cách (pixel) tới obstacle đầu tiên hoặc max_range nếu không thấy.
    Dùng bước nhỏ hơn 1 để mịn.
    """
    step = 1.0
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    dist = 0.0
    while dist < max_range:
        rx = int(round(x + dist * cos_a))
        ry = int(round(y + dist * sin_a))
        if rx < 0 or rx >= W or ry < 0 or ry >= H:
            return dist
        if occ[ry, rx]:  # obstacle
            return dist
        dist += step
    return max_range

def simulate_measurements(x, y, theta, rays=RAY_COUNT):
    """
    Trả về mảng distances (rays) cho pose (x,y,theta).
    Rays phân bố đều quanh 360 độ.
    """
    angles = [theta + 2*math.pi*i/rays for i in range(rays)]
    dists = [ray_distance_px(x, y, a) for a in angles]
    return np.array(dists), np.array(angles)

def normalize_weights(w):
    s = np.sum(w)
    if s == 0:
        return np.ones_like(w) / len(w)
    return w / s

def systematic_resample(weights):
    N = len(weights)
    positions = (np.arange(N) + np.random.random()) / N
    indexes = np.zeros(N, 'i')
    cumulative_sum = np.cumsum(weights)
    i, j = 0, 0
    while i < N:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    return indexes

# ======== Particle Filter trạng thái ========
particles = None   # mảng shape (N,3): x,y,theta
weights = None
robot_pose = None  # (x,y,theta) hoặc None nếu chưa đặt
running_auto = False

def init_particles(n=NUM_PARTICLES):
    global particles, weights
    pts = np.zeros((n, 3), dtype=np.float32)
    for i in range(n):
        x, y = random_free_point()
        th = np.random.rand() * 2*math.pi
        pts[i] = (x, y, th)
    particles = pts
    weights = np.ones(n, dtype=np.float32) / n

def predict(particles, vx=0.0, vy=0.0, dtheta=0.0):
    # Trong demo này, ta mô phỏng 1 bước predict nhỏ với motion noise.
    N = particles.shape[0]
    # apply small random motion (robot chủ động không di chuyển trừ khi muốn)
    particles[:,0] += vx + np.random.randn(N) * MOTION_NOISE_POS
    particles[:,1] += vy + np.random.randn(N) * MOTION_NOISE_POS
    particles[:,2] += dtheta + np.random.randn(N) * MOTION_NOISE_ANGLE
    # giữ trong range
    particles[:,2] = (particles[:,2] + math.pi*2) % (2*math.pi)
    # nếu particle rơi vào obstacle hoặc ra ngoài, đặt lại ngẫu nhiên
    for i in range(N):
        px = int(round(particles[i,0])); py = int(round(particles[i,1]))
        if px < 0 or px >= W or py < 0 or py >= H or occ[py, px]:
            x,y = random_free_point()
            particles[i,0] = x; particles[i,1] = y; particles[i,2] = np.random.rand()*2*math.pi

def update_weights(particles, robot_measurements):
    """
    robot_measurements: 1D array length RAY_COUNT (đã có noise)
    cho mỗi particle: mô phỏng measurements và tính likelihood Gaussian.
    """
    global weights
    N = particles.shape[0]
    R = len(robot_measurements)
    # tốc độ: vector hoá bằng vòng lặp trên particles (R không quá lớn)
    w = np.zeros(N, dtype=np.float64)
    for i in range(N):
        px, py, pth = particles[i]
        sim_dists, _ = simulate_measurements(px, py, pth, rays=R)
        # lỗi bình phương
        err = robot_measurements - sim_dists
        # likelihood (gaussian độc lập trên từng tia)
        prob = np.exp(-0.5 * (err*err) / (SENSOR_NOISE**2))
        # kết hợp (số nhân), nhưng để tránh underflow, dùng log sum? ở đây nhỏ R nên ok
        w[i] = np.prod(prob) + 1e-300
    weights[:] = normalize_weights(w)

def resample_particles():
    global particles, weights
    idx = systematic_resample(weights)
    particles = particles[idx].copy()
    weights = np.ones_like(weights) / len(weights)

# ======== Mouse callback ========
def mouse_cb(event, mx, my, flags, param):
    global robot_pose
    if event == cv2.EVENT_LBUTTONDOWN:
        # đặt robot tại vị trí click, orientation = 0 (pointing to +x)
        # Lưu theo pixel coords
        robot_pose = (mx, my, 0.0)
        print(f"Robot set to pixel: ({mx}, {my})")
    # có thể thêm right click để đặt heading nếu muốn

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.setMouseCallback(WINDOW_NAME, mouse_cb)

# Khởi tạo particles ban đầu
init_particles()

last_step_time = time.time()
auto_delay = 0.05  # s between auto steps

print("Hướng dẫn: click trái -> đặt robot; 'l' init particles; 's' step; 'a' auto toggle; 'r' reset robot; 'q' quit")

# ======== Vòng lặp hiển thị chính ========
while True:
    display = orig.copy()

    # Vẽ particles
    if particles is not None:
        for i in range(len(particles)):
            px = int(round(particles[i,0])); py = int(round(particles[i,1]))
            if 0 <= px < W and 0 <= py < H:
                # alpha theo weight (nếu weights tồn tại)
                c = (200, 50, 200)
                if weights is not None:
                    alpha = int(255 * (weights[i] / (weights.max()+1e-12)))
                    # làm màu sáng hơn theo weight
                    col = (int(50 + alpha*0.7), 50, int(255 - alpha*0.6))
                    cv2.circle(display, (px, py), 2, col, -1)
                else:
                    cv2.circle(display, (px, py), 2, c, -1)

    # Vẽ robot đo (nếu có)
    if robot_pose is not None:
        rx, ry, rth = robot_pose
        cv2.circle(display, (int(rx), int(ry)), 6, (0, 0, 255), -1)
        # vẽ measurement rays
        dists, angles = simulate_measurements(rx, ry, rth, rays=RAY_COUNT)
        for dd, aa in zip(dists, angles):
            ex = int(round(rx + dd * math.cos(aa)))
            ey = int(round(ry + dd * math.sin(aa)))
            cv2.line(display, (int(rx), int(ry)), (ex, ey), (0,255,255), 1)

    cv2.imshow(WINDOW_NAME, display)
    key = cv2.waitKey(1) & 0xFF

    # tự động chạy nếu bật
    if running_auto and (time.time() - last_step_time) > auto_delay:
        key = ord('s')
        last_step_time = time.time()

    if key == ord('q'):
        break
    elif key == ord('l'):
        init_particles()
        print("Particles initialized.")
    elif key == ord('r'):
        robot_pose = None
        print("Robot reset.")
    elif key == ord('a'):
        running_auto = not running_auto
        print("Auto running:", running_auto)
    elif key == ord('s'):
        # 1 step particle filter: predict -> update -> resample
        if robot_pose is None:
            print("Set robot pose first (left click).")
            continue
        # predict (no actual commanded motion in this demo; we add small random walk)
        predict(particles, vx=0.0, vy=0.0, dtheta=0.0)
        # get robot measurement (with noise)
        rx, ry, rth = robot_pose
        robot_dists, _ = simulate_measurements(rx, ry, rth, rays=RAY_COUNT)
        # add sensor noise to "observed" distances
        robot_dists_noisy = robot_dists + np.random.randn(len(robot_dists)) * SENSOR_NOISE
        # update weights
        update_weights(particles, robot_dists_noisy)
        # resample
        resample_particles()
        print("Step done. Particle weights updated & resampled.")

cv2.destroyAllWindows()
