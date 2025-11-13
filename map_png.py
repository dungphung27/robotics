import cv2
import numpy as np

# ---- Thông số phòng ----
ROOM_SIZE = 6.0  # mét
SCALE = 100       # 1m = 100 px
IMG_SIZE = int(ROOM_SIZE * SCALE)

# ---- Danh sách chướng ngại vật ----
OBSTACLES = [
    {"x": 2.0, "y": 2.0, "r": 0.4},
    {"x": 4.0, "y": 3.0, "r": 0.5},
    {"x": 3.0, "y": 5.0, "r": 0.3},
]

# ---- Tạo nền trắng ----
img = np.ones((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8) * 255

# ---- Vẽ viền phòng ----
cv2.rectangle(img, (0, 0), (IMG_SIZE - 1, IMG_SIZE - 1), (0, 0, 0), 2)

# ---- Vẽ chướng ngại vật ----
for obs in OBSTACLES:
    cx = int(obs["x"] * SCALE)
    cy = int(ROOM_SIZE * SCALE - obs["y"] * SCALE)  # lật trục y
    r = int(obs["r"] * SCALE)
    cv2.circle(img, (cx, cy), r, (40, 40, 40), -1)  # màu xám đậm

# ---- Hiển thị & lưu ----
cv2.imshow("Room Map (Clean)", img)
cv2.imwrite("room_map_clean.png", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("✅ Đã lưu bản đồ sạch vào file 'room_map_clean.png'")
