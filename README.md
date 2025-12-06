# 🎭 Sentiment Analysis - Phân tích cảm xúc tiếng Việt

Ứng dụng web phân tích cảm xúc văn bản tiếng Việt sử dụng mô hình PhoBERT - mô hình ngôn ngữ được huấn luyện đặc biệt cho tiếng Việt.

> **Đồ án môn học: Seminar chuyên đề (ngành CNTT, ngành KTPM)**
>
> - **Sinh viên:** Nguyễn Danh Đạt
> - **Mã SV:** 3122410070

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![Transformers](https://img.shields.io/badge/Transformers-4.35+-yellow.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 📖 Mô tả

Đây là một ứng dụng phân tích cảm xúc (Sentiment Analysis) cho văn bản tiếng Việt. Ứng dụng sử dụng mô hình **PhoBERT** đã được fine-tune cho tác vụ phân loại cảm xúc, có khả năng nhận diện 3 loại cảm xúc:

| Cảm xúc | Mô tả | Màu hiển thị |
|---------|-------|--------------|
| 🔵 **TÍCH CỰC** | Văn bản thể hiện cảm xúc vui vẻ, hài lòng, yêu thích | Xanh dương |
| ⚫ **TRUNG LẬP** | Văn bản không thể hiện rõ cảm xúc tích cực hay tiêu cực | Đen |
| 🔴 **TIÊU CỰC** | Văn bản thể hiện cảm xúc buồn, tức giận, không hài lòng | Đỏ |

---

## ✨ Tính năng

### 🔍 Phân tích cảm xúc
- Phân loại văn bản tiếng Việt thành 3 nhóm cảm xúc
- Hiển thị độ tin cậy (confidence score) của kết quả
- Hỗ trợ văn bản viết tắt và không dấu

### 🔤 Tiền xử lý văn bản thông minh
- Tự động thêm dấu tiếng Việt cho văn bản không dấu
- Mở rộng từ viết tắt phổ biến (vd: "k" → "không", "dc" → "được")
- Hỗ trợ ngôn ngữ Gen Z

### 💾 Lưu trữ lịch sử
- Lưu tất cả kết quả phân tích vào database SQLite
- Xem lại lịch sử 50 bản ghi gần nhất
- Hiển thị với màu sắc trực quan

### 📋 JSON Output
- Xuất kết quả dưới dạng JSON chuẩn
- Dễ dàng tích hợp với các hệ thống khác

### 🎯 Fine-tuning Model
- Hỗ trợ huấn luyện lại model với dữ liệu tùy chỉnh
- Script training tự động

---

## 🚀 Hướng dẫn sử dụng

### 1. Cài đặt

```bash
# Clone repository
git clone https://github.com/your-username/SentimentAnalysisProject.git
cd SentimentAnalysisProject

# Tạo môi trường ảo
python -m venv .venv

# Kích hoạt môi trường ảo
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Cài đặt dependencies
pip install -r requirements.txt
```

### 2. Chạy ứng dụng

```bash
streamlit run app.py
```

Ứng dụng sẽ mở tại: `http://localhost:8501`

### 3. Sử dụng

1. **Nhập văn bản** vào ô input (ít nhất 5 ký tự)
2. **Nhấn "Phân loại cảm xúc"** để phân tích
3. **Xem kết quả** với màu sắc và độ tin cậy
4. **Xem JSON Output** bằng cách mở expander

### 4. Huấn luyện model (tùy chọn)

```bash
python training_model.py
```

### 5. Chạy test

```bash
python run_tests.py
```

---

## 🛠️ Công nghệ sử dụng

| Công nghệ | Phiên bản | Mục đích |
|-----------|-----------|----------|
| **Python** | 3.8+ | Ngôn ngữ lập trình chính |
| **Streamlit** | 1.28+ | Framework xây dựng giao diện web |
| **Transformers** | 4.35+ | Thư viện NLP của Hugging Face |
| **PhoBERT** | - | Mô hình ngôn ngữ tiếng Việt |
| **PyTorch** | 2.0+ | Deep learning framework |
| **SQLite** | - | Cơ sở dữ liệu lưu trữ lịch sử |
| **Pandas** | - | Xử lý và hiển thị dữ liệu |

### 📦 Model sử dụng

- **[wonrax/phobert-base-vietnamese-sentiment](https://huggingface.co/wonrax/phobert-base-vietnamese-sentiment)**: PhoBERT đã được fine-tune cho tác vụ phân tích cảm xúc tiếng Việt

---

## 📁 Cấu trúc dự án

```
SentimentAnalysisProject/
├── 📄 app.py                 # Ứng dụng Streamlit chính
├── 📄 train_model.py         # Script huấn luyện model
├── 📄 connect_db.py          # Tiện ích kết nối database
├── 📄 requirements.txt       # Dependencies
├── 📄 README.md              # Tài liệu hướng dẫn               
├── 📄 run_tests.py           # Script chạy test
├── 📄 test_cases.json        # Các test case
│
├── 📂 tuned_model/           # Model đã fine-tune (nếu có)
│   ├── 📄 config.json
│   ├── 📄 model.safetensors
│   ├── 📄 tokenizer_config.json
│   └── 📄 vocab.txt
│
├── 📂 .cache/                # Cache model Hugging Face
│
└── 📄 sentiments.db          # Database SQLite (tự tạo khi chạy)
```

---

## 📊 Ví dụ sử dụng

### Input & Output

| Input | Output | Độ tin cậy |
|-------|--------|------------|
| "Tôi rất vui hôm nay" | 🔵 TÍCH CỰC | 0.98 |
| "Món ăn này dở quá" | 🔴 TIÊU CỰC | 0.95 |
| "Bây giờ là 7 giờ sáng" | ⚫ TRUNG LẬP | 0.87 |
| "k thich cai nay" | 🔴 TIÊU CỰC | 0.92 |

### JSON Output

```json
{
    "text": "Tôi rất vui hôm nay",
    "sentiment": "positive",
    "score": 0.9823
}
```

---

## 🔧 Cấu hình

### Biến môi trường

| Biến | Mô tả | Mặc định |
|------|-------|----------|
| `TRANSFORMERS_CACHE` | Thư mục cache model | `.cache/` |
| `HF_HOME` | Thư mục Hugging Face | `.cache/` |

### Tùy chỉnh

- **Model**: Thay đổi biến `MODEL_NAME` trong `app.py`
- **Database**: Thay đổi biến `DB_PATH` trong `app.py`
- **Số lượng lịch sử**: Thay đổi tham số `limit` trong `get_sentiment_history()`

---

## 📝 Từ viết tắt được hỗ trợ

| Viết tắt | Đầy đủ | Viết tắt | Đầy đủ |
|----------|--------|----------|--------|
| k, ko | không | dc, đc | được |
| cx | cũng | j | gì |
| mk, mik | mình | bn | bạn |
| bt, bik | biết | vs | với |
| r | rồi | v | vậy |
| tks, thanks | cảm ơn | sr, sorry | xin lỗi |

## 📄 License

Dự án được phân phối dưới giấy phép MIT. Xem file `LICENSE` để biết thêm chi tiết.

---

## 👨‍💻 Tác giả

**Nguyễn Danh Đạt**

---

## 🙏 Cảm ơn

- [Hugging Face](https://huggingface.co/) - Thư viện Transformers
- [VinAI Research](https://github.com/VinAIResearch/PhoBERT) - PhoBERT model
- [Streamlit](https://streamlit.io/) - Framework xây dựng web app