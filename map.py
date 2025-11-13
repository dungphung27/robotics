import serial
import zlib
import threading
import numpy as np
import cv2
import math
from robot import Robot
from particle_filter import ParticleFilter


# ==========================
#  CẤU HÌNH SERIAL
# ==========================
ser = serial.Serial('/dev/opencr', 460800)


# ==========================
#  HÀM ĐỌC DỮ LIỆU SERIAL (CÓ CRC-32)
# ==========================
def read_csv_packet():
    """Đọc 1 gói dữ liệu có CRC32, trả về (v, góc_rad, lidar_list)"""
    line = ser.readline().decode('utf-8').strip()
    if not line:
        return None, None, None

    fields = line.split(',')
    if len(fields) < 8:
        print("⚠️ Gói dữ liệu quá ngắn:", fields)
        return None, None, None

    try:
        crc_recv = int(fields[-1], 16)
    except ValueError:
        print("⚠️ CRC không hợp lệ hoặc thiếu!")
        return None, None, None

    # Kiểm tra CRC
    payload_fields = fields[:-1]
    payload_bytes = ",".join(payload_fields).encode('utf-8')
    crc_calc = zlib.crc32(payload_bytes) & 0xFFFFFFFF

    if crc_calc != crc_recv:
        print(f"❌ CRC lỗi! tính được {hex(crc_calc)}, nhận {hex(crc_recv)}")
        return None, None, None

    try:
        # ---- ĐỌC DỮ LIỆU CHÍNH ----
        v_val = float(fields[1])  # tốc độ (m/s)
        angle_deg = float(fields[3])  # góc tuyệt đối (độ)
        angle_rad = math.radians(angle_deg)  # đổi sang radian

        # ---- LIDAR ----
        lidar_values = []
        for x in fields[6:-2]:
            try:
                lidar_values.append(int(x) / 1000.0)  # mm → m
            except ValueError:
                continue

        return v_val, angle_rad, lidar_values

    except Exception as e:
        print(f"⚠️ Parse lỗi: {e}")
        return None, None, None


# ==========================
#  CLASS MAPPER
# ==========================
class LidarMapper:
    def __init__(self, map_size=800, scale=50, dt=0.1, img_path=None, mode="fast"):
        self.MAP_SIZE = np.array([map_size, map_size], dtype=int)
        self.SCALE = scale
        self.CENTER = np.array([map_size // 2, map_size // 2], dtype=int)
        self.dt = dt
        self.mode = mode  # "normal" hoặc "fast"

        # Nền bản đồ
        if img_path:
            img = cv2.imread(img_path)
            if img is None:
                raise FileNotFoundError(f"Không tìm thấy ảnh: {img_path}")
            img = cv2.resize(img, (300, 300))
            self.map_img = img.copy()
            self.MAP_SIZE = np.array([img.shape[0], img.shape[1]], dtype=int)
            self.CENTER = np.array([self.MAP_SIZE[1] // 2, self.MAP_SIZE[0] // 2], dtype=int)
        else:
            self.map_img = np.ones((map_size, map_size, 3), dtype=np.uint8) * 128

        self.clean_map = self.map_img.copy()
        self.x, self.y, self.theta = 0.0, 0.0, 0.0
        self.path = []

        # --- Bộ nhớ điểm vật cản ---
        self.obstacle_confirm = {}  # {(px, py): số_lần_phát_hiện}
        self.confirm_threshold = 3 if mode == "fast" else 5
        self.pixel_reduce = 2 if mode == "fast" else 1  # Giảm độ phân giải khi lưu điểm
        self.max_obstacles = 40000  # Giới hạn bộ nhớ

        cv2.namedWindow("Lidar Map", cv2.WINDOW_NORMAL)

    def expand_map_if_needed(self, px, py):
        h, w = self.map_img.shape[:2]
        expand = False
        top = bottom = left = right = 0
        if px < 0:
            left = abs(px) + 100; expand = True
        elif px >= w:
            right = px - w + 100; expand = True
        if py < 0:
            top = abs(py) + 100; expand = True
        elif py >= h:
            bottom = py - h + 100; expand = True

        if expand:
            new_h = int(h + top + bottom)
            new_w = int(w + left + right)
            new_map = np.ones((new_h, new_w, 3), dtype=np.uint8) * 128
            new_map[int(top):int(top + h), int(left):int(left + w)] = self.map_img
            new_clean = np.ones((new_h, new_w, 3), dtype=np.uint8) * 128
            new_clean[int(top):int(top + h), int(left):int(left + w)] = self.clean_map
            self.map_img = new_map
            self.clean_map = new_clean
            self.MAP_SIZE = np.array([new_h, new_w], dtype=int)
            print(f"🟢 Mở rộng bản đồ: {new_w}x{new_h}")

    def draw_lidar(self, lidar):
        """Vẽ LIDAR với xác nhận vật cản (tối ưu hiệu năng)"""
        angles = np.deg2rad(np.arange(len(lidar)))
        cos_a = np.cos(angles)
        sin_a = np.sin(angles)
        rx = int(self.CENTER[0] + self.x * self.SCALE)
        ry = int(self.CENTER[1] - self.y * self.SCALE)

        for i, dist in enumerate(lidar):
            if dist <= 0.02:  # bỏ nhiễu cực gần
                continue

            x_end = self.x + dist * cos_a[i]
            y_end = self.y + dist * sin_a[i]
            ex = int(self.CENTER[0] + x_end * self.SCALE)
            ey = int(self.CENTER[1] - y_end * self.SCALE)

            # Tự mở rộng bản đồ nếu cần
            self.expand_map_if_needed(ex, ey)

            # Giảm số lần vẽ line (chỉ vẽ 1/3 số tia để tiết kiệm CPU)
            if i % 3 == 0:
                cv2.line(self.map_img, (rx, ry), (ex, ey), (255, 255, 255), 1)

            # Lọc xác nhận vật cản
            key = (ex // self.pixel_reduce, ey // self.pixel_reduce)
            self.obstacle_confirm[key] = self.obstacle_confirm.get(key, 0) + 1
            if self.obstacle_confirm[key] >= self.confirm_threshold:
                cv2.circle(self.map_img, (ex, ey), 2, (0, 0, 0), -1)

        # Giới hạn bộ nhớ
        if len(self.obstacle_confirm) > self.max_obstacles:
            self.obstacle_confirm.clear()

    def update_robot_state(self, v, angle):
        """Cập nhật vị trí robot"""
        self.theta = angle
        self.x += v * math.cos(self.theta) * self.dt
        self.y -= v * math.sin(self.theta) * self.dt
        self.path.append((self.x, self.y))

    def draw_robot(self):
        rx = int(self.CENTER[0] + self.x * self.SCALE)
        ry = int(self.CENTER[1] - self.y * self.SCALE)
        hx = int(rx + 15 * math.cos(self.theta))
        hy = int(ry + 15 * math.sin(self.theta))
        cv2.circle(self.map_img, (rx, ry), 6, (0, 0, 255), -1)
        cv2.line(self.map_img, (rx, ry), (hx, hy), (0, 0, 0), 2)

    def draw_path(self):
        if len(self.path) < 2:
            return
        pts = np.array([
            [int(self.CENTER[0] + x * self.SCALE),
             int(self.CENTER[1] - y * self.SCALE)]
            for (x, y) in self.path
        ], np.int32)
        cv2.polylines(self.map_img, [pts], False, (0, 255, 0), 2)




# ==========================
#  THREAD CHÍNH
# ==========================
mapper = LidarMapper()
robot = Robot()
pf = ParticleFilter(num_particles=200, map_img=mapper.map_img, initial_pos=None)

def serial_thread():
    while True:
        v, angle, lidar = read_csv_packet()
        if v is None:
            continue

        mapper.update_robot_state(v, angle)
        mapper.draw_lidar(lidar)
        mapper.draw_path()
        mapper.draw_robot()

        display = mapper.map_img.copy()
        cv2.imshow("Lidar Map", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def keyboard_thread():
    while True:
        cmd = input("Nhập lệnh: ")
        if cmd.lower() == "exit":
            print("Thoát chương trình...")
            ser.close()
            cv2.destroyAllWindows()
            exit(0)
        else:
            ser.write((cmd + "\n").encode('utf-8'))
            print(f"📤 Đã gửi: {cmd}")


# ==========================
#  KHỞI CHẠY SONG SONG
# ==========================
t1 = threading.Thread(target=serial_thread, daemon=True)
t2 = threading.Thread(target=keyboard_thread, daemon=True)
t1.start()
t2.start()
t1.join()
t2.join()
