import serial
import zlib
import threading
import numpy as np
import cv2
import math
from robot import Robot
from particle_filter import ParticleFilter
import time
from collections import OrderedDict


# ==========================
#  CẤU HÌNH SERIAL
# ==========================
ser = serial.Serial('/dev/opencr', 460800)


# ==========================
#  HÀM ĐỌC DỮ LIỆU SERIAL (CÓ CRC-32)
# ==========================
def read_csv_packet():
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

    payload_fields = fields[:-1]
    payload_bytes = ",".join(payload_fields).encode('utf-8')
    crc_calc = zlib.crc32(payload_bytes) & 0xFFFFFFFF

    if crc_calc != crc_recv:
        print(f"❌ CRC lỗi! tính được {hex(crc_calc)}, nhận {hex(crc_recv)}")
        return None, None, None

    try:
        v_val = float(fields[1])
        angle_deg = float(fields[3])
        angle_rad = math.radians(angle_deg)

        lidar_values = []
        for x in fields[6:-2]:
            try:
                lidar_values.append(int(x) / 1000.0)
            except ValueError:
                continue

        return v_val, angle_rad, lidar_values

    except Exception as e:
        print(f"⚠️ Parse lỗi: {e}")
        return None, None, None


class LidarMapper:
    def __init__(self, map_size=800, scale=50, dt=0.1, img_path=None, mode="fast"):
        self.SCALE = scale
        self.dt = dt
        self.mode = mode

        if img_path:
            img = cv2.imread(img_path)
            if img is None:
                raise FileNotFoundError(f"Không tìm thấy ảnh: {img_path}")
            img = cv2.resize(img, (300, 300))
            self.map_img = img.copy()
        else:
            self.map_img = np.ones((map_size, map_size, 3), dtype=np.uint8) * 128

        self.clean_map = self.map_img.copy()
        h, w = self.map_img.shape[:2]
        self.MAP_SIZE = np.array([h, w], dtype=int)

        self.CENTER = np.array([w // 2, h // 2], dtype=int)

        # Robot states
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        # No PATH
        # self.path = []   # xoá bỏ

        self.margin = 120
        self.expand_size = 250
        self.min_expand_interval = 1.0
        self.last_expand_time = 0.0

        self.draw_skip = 3 if mode == "fast" else 1
        self.confirm_threshold = 3 if mode == "fast" else 5
        self.pixel_reduce = 2 if mode == "fast" else 1

        self.obstacle_confirm = OrderedDict()
        self.max_obstacles = 30000

        cv2.namedWindow("Lidar Map", cv2.WINDOW_NORMAL)

    def _cap_obstacles(self):
        while len(self.obstacle_confirm) > self.max_obstacles:
            self.obstacle_confirm.popitem(last=False)

    def expand_map_if_needed(self, rx, ry):
        now = time.time()
        if now - self.last_expand_time < self.min_expand_interval:
            return

        h, w = self.map_img.shape[:2]
        top = bottom = left = right = 0
        expand = False

        if rx < self.margin:
            left = self.expand_size; expand = True
        elif rx >= w - self.margin:
            right = self.expand_size; expand = True

        if ry < self.margin:
            top = self.expand_size; expand = True
        elif ry >= h - self.margin:
            bottom = self.expand_size; expand = True

        if not expand:
            return

        new_h = h + top + bottom
        new_w = w + left + right

        new_map = np.ones((new_h, new_w, 3), dtype=np.uint8) * 128
        new_clean = np.ones((new_h, new_w, 3), dtype=np.uint8) * 128

        new_map[top:top + h, left:left + w] = self.map_img
        new_clean[top:top + h, left:left + w] = self.clean_map

        self.map_img = new_map
        self.clean_map = new_clean
        self.MAP_SIZE = np.array([new_h, new_w], dtype=int)

        self.CENTER += np.array([left, top])
        self.last_expand_time = now

        print(f"🟢 Mở rộng bản đồ: {new_w}x{new_h}")

    def draw_lidar(self, lidar):
        angles = np.deg2rad(np.arange(len(lidar)))
        cos_a = np.cos(angles)
        sin_a = np.sin(angles)

        # Robot position (float)
        rx_f = self.CENTER[0] + self.x * self.SCALE
        ry_f = self.CENTER[1] - self.y * self.SCALE

        # Convert to int only for drawing
        rx = int(rx_f)
        ry = int(ry_f)

        # Expand map if needed
        self.expand_map_if_needed(rx, ry)

        h, w = self.map_img.shape[:2]

        for i, dist in enumerate(lidar):
            if dist <= 0.02:
                continue

            # Compute hit point (float world -> float pixel)
            x_end = self.x + dist * cos_a[i]
            y_end = self.y + dist * sin_a[i]

            ex_f = self.CENTER[0] + x_end * self.SCALE
            ey_f = self.CENTER[1] - y_end * self.SCALE

            # Only convert to int for drawing & boundary check
            ex = int(ex_f)
            ey = int(ey_f)

            if not (0 <= ex < w and 0 <= ey < h):
                continue

            if i % self.draw_skip == 0:
                cv2.line(self.map_img, (rx, ry), (ex, ey), (255, 255, 255), 1)

            key = (ex // self.pixel_reduce, ey // self.pixel_reduce)
            if key in self.obstacle_confirm:
                cnt = self.obstacle_confirm.pop(key)
                self.obstacle_confirm[key] = cnt + 1
            else:
                self.obstacle_confirm[key] = 1

            if self.obstacle_confirm[key] >= self.confirm_threshold:
                cv2.circle(self.map_img, (ex, ey), 2, (0, 0, 0), -1)

                if self.obstacle_confirm[key] == self.confirm_threshold:
                    cv2.circle(self.clean_map, (ex, ey), 2, (0, 0, 0), -1)

        self._cap_obstacles()



    def update_robot_state(self, v, angle):
        if abs(v) > 0.5:
            return
        self.theta = angle
        print(f"🤖 Robot state update: v={v:.6f} m/s, θ={math.degrees(self.theta):.2f}°")
        self.x += v * math.cos(self.theta) * self.dt
        self.y -= v * math.sin(self.theta) * self.dt

        # ⚠️ XÓA BỎ PATH
        # self.path.append((self.x, self.y))

        rx = int(self.CENTER[0] + self.x * self.SCALE)
        ry = int(self.CENTER[1] - self.y * self.SCALE)
        if (rx < self.margin or ry < self.margin or
            rx >= self.MAP_SIZE[1] - self.margin or
            ry >= self.MAP_SIZE[0] - self.margin):
            self.expand_map_if_needed(rx, ry)

    def draw_robot(self):
        rx = int(self.CENTER[0] + self.x * self.SCALE)
        ry = int(self.CENTER[1] - self.y * self.SCALE)

        hx = int(rx + 15 * math.cos(self.theta))
        hy = int(ry + 15 * math.sin(self.theta))

        cv2.circle(self.map_img, (rx, ry), 6, (0, 0, 255), -1)
        cv2.line(self.map_img, (rx, ry), (hx, hy), (0, 0, 0), 2)

target_px = None

def mouse_callback(event, x, y, flags, param):
    global target_px
    if event == cv2.EVENT_LBUTTONDOWN:
        target_px = (x, y)
        print(f"📌 Click target PX = {target_px}")

# cv2.setMouseCallback("Lidar Map", mouse_callback)
class MoveToPoint:
    def __init__(self, mapper):
        self.mapper = mapper
        self.active = False
        self.tx = None
        self.ty = None

        # trạng thái quay
        self.rotating = False
        self.target_theta = None   # góc mục tiêu khi đang quay (rad)
        self.rotate_tolerance = math.radians(3.0)  # tolerance ~3 độ

        # tham số di chuyển
        self.dist_tolerance = 0.10  # 10 cm
        self.rotate_cmd_sent = False

    def set_target(self, px, py):
        # convert PX → World (theo scale và CENTER của mapper)
        self.tx = (px - self.mapper.CENTER[0]) / self.mapper.SCALE
        self.ty = -(py - self.mapper.CENTER[1]) / self.mapper.SCALE
        self.active = True
        self.rotating = False
        self.target_theta = None
        self.rotate_cmd_sent = False
        print(f"🎯 Target world: {self.tx:.2f}, {self.ty:.2f}")

    def _angle_diff(self, a, b):
        # trả về a - b trong [-pi, pi]
        d = a - b
        return math.atan2(math.sin(d), math.cos(d))

    def update(self):
        if not self.active:
            return

        # lấy trạng thái hiện tại từ mapper
        x = self.mapper.x
        y = self.mapper.y
        th = self.mapper.theta  # mapper.theta phải là góc relative->global (đã xử lý initial)
        # print("th: {th:2f} rad, {deg:.2f}°".format(th=th, deg=math.degrees(th)))
        dx = self.tx - x
        dy = self.ty - y
        dist = math.hypot(dx, dy)
        goal_ang = math.atan2(-dy, dx)
        print(f"➡ Target angle: {math.degrees(goal_ang):.2f}")
        # tính sai lệch góc cần xoay (goal_ang - current)
        dth = self._angle_diff(goal_ang, th)
        print("Δth:", f"{math.degrees(dth):.2f}°")
        # ---------- PHASE 1: CHƯA XOAY, cần gửi lệnh xoay 1 lần ----------
        if not self.rotating:
            if abs(dth) > 0.15:  # > ~8.6 độ -> cần xoay
                deg = int(round(abs(math.degrees(dth))))
                # Hạn chế deg tối đa (nếu cần)
                if deg == 0:
                    deg = 1

                # Nếu dth > 0 => goal nằm CCW so với heading hiện tại.
                # Theo bạn: 'aXX' = xoay trái (CCW), 'dXX' = xoay phải (CW)
                if dth > 0:
                    cmd = f"d{deg}"
                    # target_theta = current + deg (rad)
                    self.target_theta = th + math.radians(deg)
                else:
                    cmd = f"a{deg}"
                    self.target_theta = th - math.radians(deg)

                # normalize target_theta về [-pi,pi]
                self.target_theta = math.atan2(math.sin(self.target_theta), math.cos(self.target_theta))

                # gửi lệnh xoay *một lần duy nhất*
                ser.write((cmd + "\n").encode())
                print("🔄 Sent rotate:", cmd, f" expect theta ~ {math.degrees(self.target_theta):.1f}°")
                self.rotating = True
                self.rotate_cmd_sent = True
                return
            # nếu không cần xoay (góc đã đủ nhỏ) -> tiếp sang di chuyển
        else:
            # ---------- PHASE 2: Đang chờ robot xoay tới target_theta ----------
            # Nếu vẫn chưa set target_theta (vì deg rounding) thì fallback kiểm dth nhỏ
            if self.target_theta is not None:
                err = self._angle_diff(self.target_theta, th)
                # nếu chưa đạt, đợi — không gửi lệnh quay lại
                if abs(err) > self.rotate_tolerance:
                    # chờ robot hoàn tất xoay (board sẽ cập nhật angle)
                    return
                else:
                    # đã xoay xong
                    print(f"✅ Rotate done. current theta: {math.degrees(th):.2f}°")
                    self.rotating = False
                    self.rotate_cmd_sent = False
            else:
                # nếu vì lý do nào đó không có target_theta, fallback dùng dth
                if abs(dth) > 0.15:
                    return
                else:
                    self.rotating = False

        # ---------- PHASE 3: Di chuyển tiến về mục tiêu ----------
        if dist > self.dist_tolerance:
            # trước khi gửi 'w', bạn có thể check thêm obstacle bằng lidar/clean_map nếu muốn
            ser.write(b"w\n")
            print("⬆ Moving w, dist:", f"{dist:.2f} m")
            return

        # ---------- PHASE 4: Đến nơi ----------
        ser.write(b"x\n")
        print("🏁 Reached target!")
        self.active = False
        self.rotating = False
        self.target_theta = None
        self.rotate_cmd_sent = False


# ==========================
#  THREAD CHÍNH
# ==========================
mapper = LidarMapper()
robot = Robot()
pf = ParticleFilter(num_particles=200, map_img=mapper.map_img, initial_pos=None)
mover = MoveToPoint(mapper)
def serial_thread():
    while True:
        v, angle, lidar = read_csv_packet()
        if v is None:
            continue

        mapper.update_robot_state(v, angle)
        mapper.draw_lidar(lidar)
        mapper.draw_robot()

        display = mapper.map_img.copy()
        global target_px
        if target_px is not None:
            mover.set_target(target_px[0], target_px[1])
            target_px = None

        mover.update()
        cv2.imshow("Lidar Map", display)
        cv2.setMouseCallback("Lidar Map", mouse_callback)
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


t1 = threading.Thread(target=serial_thread, daemon=True)
t2 = threading.Thread(target=keyboard_thread, daemon=True)
t1.start()
t2.start()
t1.join()
t2.join()
