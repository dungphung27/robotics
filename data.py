import cv2
import numpy as np
import json
import math

# --- Cấu hình ---
IMG_PATH = "test.png"
PX_PER_M = 100                 # 1m = 100 px
LIDAR_RANGE_MIN = 0.12
LIDAR_RANGE_MAX = 3.5
LIDAR_NUM_POINTS = 360
DT = 0.1
NUM_PACKETS = 500
V_LINEAR = 0.25                # m/s
TURN_SPEED = math.pi / 2       # rad/s
LIDAR_NOISE = 0.02

# --- Đọc ảnh bản đồ ---
img = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("Không tìm thấy ảnh bản đồ!")

h, w = img.shape
print(f"✅ Kích thước ảnh: {w}x{h}")

# --- Hàm bắn 1 tia Lidar ---
def cast_ray(x, y, angle):
    dx, dy = math.cos(angle), math.sin(angle)
    for step in range(int(LIDAR_RANGE_MAX * PX_PER_M)):
        px = int(x + dx * step)
        py = int(y + dy * step)
        if px < 0 or py < 0 or px >= w or py >= h:
            break
        if img[py, px] < 128:  # gặp vật cản
            dist = step / PX_PER_M
            if LIDAR_RANGE_MIN <= dist <= LIDAR_RANGE_MAX:
                return round(dist + np.random.normal(0, LIDAR_NOISE), 3)
            else:
                return 0
    return 0

# --- Sinh dữ liệu LiDAR ---
def generate_lidar(x, y, theta):
    angles = np.linspace(0, 2 * np.pi, LIDAR_NUM_POINTS, endpoint=False)
    lidar = []
    for a in angles:
        d = cast_ray(x, y, a + theta)
        lidar.append(d)
    return lidar

# --- Khởi tạo ---
x, y = w // 5, h // 2
theta = 0.0
v = V_LINEAR
omega = 0.0
phase = 0  # theo dõi hướng quay

# --- Giả lập ---
packets = []
for t in np.arange(0, NUM_PACKETS * DT, DT):
    # Sinh dữ liệu lidar
    lidar = generate_lidar(x, y, theta)

    # Ghi lại dữ liệu
    packets.append({
        "time": round(t, 2),
        "lidar": lidar,
        "x": round(x / PX_PER_M, 3),
        "y": round(y / PX_PER_M, 3),
        "theta": round(theta, 3),
        "v": round(v, 3),
        "omega": round(omega, 3),
        "phase": phase
    })

    # --- Di chuyển ---
    front_dist = cast_ray(x, y, theta)  # đo khoảng cách phía trước

    if omega == 0:  # đang đi thẳng
        # Nếu có vật cản trước mặt, dừng lại và quay
        if 0 < front_dist < 0.25:
            v = 0
            omega = TURN_SPEED
        else:
            # tiếp tục đi thẳng
            x += v * math.cos(theta) * PX_PER_M * DT
            y += v * math.sin(theta) * PX_PER_M * DT
    else:
        # Đang quay
        theta += omega * DT
        # Khi quay đủ 90 độ thì dừng quay và đi tiếp
        if abs(theta % (2 * math.pi) - (phase + 1) * math.pi / 2) < 0.05:
            omega = 0
            v = V_LINEAR
            phase += 1

# --- Lưu dữ liệu ---
with open("lidar_trajectory_data.json", "w") as f:
    json.dump(packets, f, indent=2)

print("✅ Đã tạo xong dữ liệu LiDAR -> lidar_trajectory_data.json")