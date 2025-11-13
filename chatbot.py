import json
import requests

# ===== Đọc dữ liệu từ JSON =====
with open("data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Ghép toàn bộ nội dung làm "corpus"
corpus = "\n".join([doc["content"] for doc in data])

# ===== Câu hỏi người dùng =====
query = "Trình tự thao tác đăng ký"

# ===== Tạo prompt cho mô hình =====
prompt = f"""
Dưới đây là thông tin về Trường Đại học Bách khoa Đà Nẵng (DUT):

{corpus}

Câu hỏi: {query}

Hãy trả lời chính xác, ngắn gọn dựa trên dữ liệu trên.
"""

# ===== Gửi yêu cầu đến Ollama API =====
url = "http://localhost:11434/api/generate"
payload = {
    "model": "gemma3:1b",
    "prompt": prompt,
    "stream": False
}

response = requests.post(url, json=payload)
result = response.json()

# ===== In kết quả =====
print("Câu trả lời:")
print(result["response"])
