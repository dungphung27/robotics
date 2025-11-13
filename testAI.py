import os
import subprocess
import PyPDF2
from sentence_transformers import SentenceTransformer
import chromadb

# === 1️⃣ Đọc nội dung PDF ===
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text() + "\n"
    return text

# === 2️⃣ Chia nhỏ văn bản ===
def chunk_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# === 3️⃣ Tạo hoặc tải lại vector DB ===
def build_or_load_db(pdf_path, persist_dir="db_storage"):
    # Dùng thư mục cố định để lưu Chroma database
    chroma_client = chromadb.PersistentClient(path=persist_dir)
    collection_name = "pdf_docs"

    # Nếu DB đã tồn tại, chỉ cần nạp lại
    existing_collections = [c.name for c in chroma_client.list_collections()]
    if collection_name in existing_collections:
        print(f"📦 Đang tải lại database đã lưu từ: {persist_dir}")
        return chroma_client.get_collection(collection_name)

    # Nếu chưa có thì tạo mới
    print("🚀 Đang tạo vector database mới...")
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    collection = chroma_client.create_collection(name=collection_name)

    for i, chunk in enumerate(chunks):
        embedding = model.encode(chunk).tolist()
        collection.add(
            ids=[f"chunk_{i}"],
            embeddings=[embedding],
            documents=[chunk]
        )

    print(f"✅ Đã lưu {len(chunks)} đoạn văn vào vector database trong {persist_dir}")
    return collection

# === 4️⃣ Hỏi Ollama ===
def query_with_ollama(question, collection, model_name="gemma3:1b"):
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = embed_model.encode(question).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=3)

    context = "\n".join(results["documents"][0])

    prompt = f"""
    Dưới đây là nội dung từ tài liệu PDF:

    {context}

    Hãy dựa trên tài liệu này để trả lời câu hỏi sau bằng tiếng Việt, ngắn gọn và chính xác:
    {question}
    """

    print("\n🤖 Đang hỏi mô hình Ollama...\n")

    result = subprocess.run(
        ["ollama", "run", model_name],
        input=prompt.encode("utf-8"),
        capture_output=True
    )

    print("📜 Trả lời:")
    print(result.stdout.decode("utf-8"))

# === 5️⃣ Chạy chương trình chính ===
if __name__ == "__main__":
    pdf_path = "dut.pdf"  # 👉 Thay bằng tên file PDF của bạn
    persist_dir = "db_storage"  # 📁 Thư mục lưu database

    if not os.path.exists(pdf_path):
        print(f"❌ Không tìm thấy file PDF: {pdf_path}")
        exit()

    collection = build_or_load_db(pdf_path, persist_dir=persist_dir)

    while True:
        question = input("\n❓ Nhập câu hỏi của bạn (hoặc gõ 'exit' để thoát): ")
        if question.lower() == "exit":
            break
        query_with_ollama(question, collection)
