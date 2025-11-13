import numpy as np
import json
import math

# --- Thông số ---
dt = 0.1  # 100ms
num_packets = 1000
room_width = 5.0
room_height = 4.0
wall_thickness = 0.3
lidar_noise = 0.02
lidar_num_points = 360

# --- Thêm chướng ngại vật ---
# Mỗi vật là hình chữ nhật: (x_min, y_min, x_max, y_max)
obstacles = [
    {"x": 2.0, "y": 1.5, "r": 0.3},
    {"x": 3.8, "y": 2.5, "r": 0.4},
    {"x": 1.2, "y": 3.0, "r": 0.25},
]

# --- Hàm tính khoảng cách tia tới vật tròn ---
def ray_circle_distance(x, y, dx, dy, cx, cy, r):
    # vector từ robot -> tâm vật
    ox, oy = cx - x, cy - y
    t_ca = ox * dx + oy * dy  # độ dài chiếu lên tia
    if t_ca < 0:
        return None  # vật ở phía sau
    d2 = ox**2 + oy**2 - t_ca**2
    if d2 > r**2:
        return None  # tia không cắt vật
    t_hc = math.sqrt(r**2 - d2)
    t = t_ca - t_hc  # điểm cắt gần nhất
    if t < 0:
        return None
    return t


# --- Hàm tạo Lidar ---
def generate_lidar(x, y, theta):
    angles = np.deg2rad(np.arange(lidar_num_points))
    lidar = np.zeros_like(angles)

    for i, ang in enumerate(angles):
        a = ang + theta
        dx, dy = np.cos(a), np.sin(a)
        dist = 10  # xa nhất mặc định

        # --- Kiểm tra va chạm tường ---
        if dx > 0:
            dist = min(dist, (room_width - x) / dx)
        elif dx < 0:
            dist = min(dist, (0 - x) / dx)
        if dy > 0:
            dist = min(dist, (room_height - y) / dy)
        elif dy < 0:
            dist = min(dist, (0 - y) / dy)

        # --- Kiểm tra va chạm vật tròn ---
        for obs in obstacles:
            d = ray_circle_distance(x, y, dx, dy, obs["x"], obs["y"], obs["r"])
            if d is not None and d < dist:
                dist = d

        # --- Thêm nhiễu và giới hạn đo ---
        if 0.11 <= dist <= 3.5:
            lidar[i] = dist + np.random.randn() * lidar_noise
        else:
            lidar[i] = 0
    return lidar


# --- Quỹ đạo robot ---
x, y, theta = 1.0, 1.0, 0.0
v = 0.25
omega = 0.0
phase = 0

packets = []

for t in np.arange(0, num_packets * dt, dt):
    lidar = generate_lidar(x, y, theta)

    packets.append({
        "time": round(t, 2),
        "lidar": lidar.tolist(),
        "v": round(v, 3),
        "omega": round(omega, 3)
    })

    # --- Di chuyển ---
    if omega == 0:
        x += v * np.cos(theta) * dt
        y += v * np.sin(theta) * dt

        if phase == 0 and x >= room_width - wall_thickness:
            omega = math.pi / 2
            v = 0
        elif phase == 1 and y >= room_height - wall_thickness:
            omega = math.pi / 2
            v = 0
        elif phase == 2 and x <= wall_thickness:
            omega = math.pi / 2
            v = 0
        elif phase == 3 and y <= wall_thickness:
            omega = math.pi / 2
            v = 0
    else:
        theta += omega * dt
        if abs(theta % (2 * math.pi) - phase * math.pi / 2 - math.pi / 2) < 0.05:
            omega = 0
            v = 0.25
            phase = (phase + 1) % 4

# --- Lưu file ---
with open("robot_room_data_with_obstacles.json", "w") as f:
    json.dump(packets, f, indent=2)

print("✅ Đã tạo dữ liệu robot có vật cản: robot_room_data_with_obstacles.json")