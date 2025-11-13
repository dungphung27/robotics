import cv2

# Đọc ảnh từ file
img = cv2.imread('image.png')

# Hiển thị ảnh
cv2.imshow('Ảnh gốc', img)

# Chờ người dùng nhấn phím bất kỳ
cv2.waitKey(0)

# Đóng tất cả cửa sổ
cv2.destroyAllWindows()
