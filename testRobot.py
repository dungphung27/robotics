import serial
import zlib
import threading

# Cấu hình cổng serial
ser = serial.Serial('/dev/opencr', 460800)

def read_csv_packet():
    line = ser.readline().decode('utf-8').strip()  # đọc 1 dòng CSV
    if not line:
        return None, None, None

    fields = line.split(',')

    # CRC-32 nhận cuối dòng
    try:
        crc_recv = int(fields[-1], 16)
    except ValueError:
        print("Dữ liệu không có CRC hợp lệ!")
        return None, None, None

    payload_fields = fields[:-1]

    # Tính CRC-32 trên payload (chuỗi CSV nối dấu phẩy)
    payload_bytes = ",".join(payload_fields).encode('utf-8')
    crc_calc = zlib.crc32(payload_bytes) & 0xFFFFFFFF

    if crc_calc != crc_recv:
        print(f"❌ CRC lỗi! tính được {hex(crc_calc)}, nhận {hex(crc_recv)}")
        return None, None, None

    # Parse các trường cơ bản
    try:
        v_val = float(fields[1])
        imu_val = int(fields[3])
        lidar_values = [int(x) for x in fields[6:-2] if x.isdigit()]
    except Exception as e:
        print(f"Lỗi parse dữ liệu: {e}")
        return None, None, None

    return v_val, imu_val, lidar_values

# Thread để đọc dữ liệu serial
def serial_thread():
    while True:
        v, imu, lidar = read_csv_packet()
        if v is not None:
            print(f"v={v}, imu={imu}, LIDAR 5 điểm đầu: {lidar[:5]} ...")

# Thread để gõ lệnh từ bàn phím
def keyboard_thread():
    while True:
        cmd = input("Nhập lệnh: ")
        if cmd.lower() == "exit":
            print("Thoát chương trình...")
            exit(0)
        else:
            # gửi lệnh xuống robot
            ser.write((cmd + "\n").encode('utf-8'))
            print(f"Đã gửi: {cmd}")

# Tạo và chạy thread
t1 = threading.Thread(target=serial_thread, daemon=True)
t2 = threading.Thread(target=keyboard_thread, daemon=True)
t1.start()
t2.start()

# Giữ main thread sống
t1.join()
t2.join()
