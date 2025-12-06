import streamlit as st
import sqlite3
import pandas as pd
import re
import json
from datetime import datetime, timezone
import os

# Thiết lập thư mục cache cho Hugging Face models
CACHE_DIR = os.path.join(os.path.dirname(__file__), '.cache')
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR
os.environ['HF_HOME'] = CACHE_DIR

# Model duy nhất sử dụng
MODEL_NAME = 'wonrax/phobert-base-vietnamese-sentiment'

DB_PATH = 'sentiments.db'

@st.cache_resource
def get_sentiment_classifier():
    """Tải và cache model PhoBERT phân tích cảm xúc tiếng Việt."""
    from transformers import pipeline
    
    try:
        classifier = pipeline('sentiment-analysis', model=MODEL_NAME, local_files_only=True)
        return classifier
    except Exception:
        classifier = pipeline('sentiment-analysis', model=MODEL_NAME)
        return classifier

@st.cache_resource
def get_db_connection():
    """Khởi tạo và trả về kết nối cơ sở dữ liệu SQLite."""
    connection = sqlite3.connect(DB_PATH, check_same_thread=False)
    cursor = connection.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS sentiments
                   (id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT, sentiment TEXT, timestamp TEXT)''')
    connection.commit()
    return connection

def save_sentiment_record(connection, text: str, sentiment: str):
    """Lưu kết quả phân tích cảm xúc vào cơ sở dữ liệu."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    cursor = connection.cursor()
    cursor.execute('INSERT INTO sentiments (text, sentiment, timestamp) VALUES (?, ?, ?)', (text, sentiment, timestamp))
    connection.commit()

def get_sentiment_history(connection, limit: int = 50):
    """Lấy lịch sử phân tích cảm xúc gần đây từ cơ sở dữ liệu."""
    cursor = connection.cursor()
    cursor.execute('SELECT text, sentiment, timestamp FROM sentiments ORDER BY id DESC LIMIT ?', (limit,))
    return cursor.fetchall()

def analyze_sentiment(classifier, text: str) -> tuple:
    """
    Phân tích cảm xúc của văn bản và trả về nhãn đã chuẩn hóa.
    
    Returns:
        Tuple gồm (nhãn_tiếng_việt, điểm_score, nhãn_gốc, json_output)
    """
    # Tiền xử lý văn bản
    processed_text = format_vietnamese_accents(text)
    
    try:
        result = classifier(processed_text)
        if isinstance(result, list) and len(result) > 0:
            label = result[0].get('label')
            score = result[0].get('score', 0.0)
            
            # Mapping nhãn PhoBERT
            vietnamese_label_map = {
                'POS': 'TÍCH CỰC',
                'NEG': 'TIÊU CỰC',
                'NEU': 'TRUNG LẬP',
            }
            
            english_label_map = {
                'POS': 'positive',
                'NEG': 'negative',
                'NEU': 'neutral',
            }
            
            normalized_label = vietnamese_label_map.get(label, label)
            english_label = english_label_map.get(label, label.lower())
            
            json_output = {
                "text": processed_text,
                "sentiment": english_label,
                "score": round(score, 4)
            }
            
            return normalized_label, score, label, json_output
            
    except Exception as e:
        st.error(f'Lỗi khi gọi model: {e}')
    
    default_json = {
        "text": processed_text if 'processed_text' in locals() else text,
        "sentiment": "neutral",
        "score": 0.0
    }
    return 'TRUNG LẬP', 0.0, 'NEUTRAL', default_json

def get_sentiment_color(sentiment: str) -> str:
    """Trả về màu tương ứng với cảm xúc."""
    if 'TIÊU CỰC' in sentiment:
        return '#DC3545'  # Đỏ
    elif 'TÍCH CỰC' in sentiment:
        return '#0D6EFD'  # Xanh dương
    return '#000000'  # Đen (Trung lập)

def format_sentiment_html(sentiment: str) -> str:
    """Format cảm xúc với màu sắc tương ứng."""
    color = get_sentiment_color(sentiment)
    return f'<span style="color: {color}; font-weight: bold;">{sentiment}</span>'

def format_vietnamese_accents(text: str) -> str:
    """Thêm dấu tiếng Việt cho văn bản thiếu dấu và mở rộng từ viết tắt."""
    if not text or not isinstance(text, str):
        return text
    
    replacements = {
        # --- viết tắt ---
        'k': 'không', 'ko': 'không', 'kh': 'không', 'dc': 'được', 'đc': 'được',
        'cx': 'cũng', 'ntn': 'như thế nào', 'nt': 'nhắn tin',
        'j': 'gì', 'gi': 'gì', 'bh': 'bây giờ',
        'mk': 'mình', 'mik': 'mình', 'v': 'vậy',
        'bn': 'bạn', 'tks': 'cảm ơn', 'thanks': 'cảm ơn',
        'bik': 'biết', 'bt': 'biết', 'sr': 'xin lỗi', 'sorry': 'xin lỗi',
        'vs': 'với', 'ok': 'được', 'okie': 'được',
        'r': 'rồi', 'nka': 'nhà',
        'ny': 'này', 'oy': 'này',
        'nko': 'nhỉ', 'h': 'giờ',
        'wa': 'quá', 'qá': 'quá', 'lm': 'làm', 'ms': 'mới',
        'trc': 'trước', 'sau': 'sau',

        # --- từ không dấu ---
        'rat': 'rất', 'vui': 'vui', 'hom': 'hôm',
        'nay': 'nay', 'toi': 'tôi', 'buon': 'buồn',
        'do': 'dở', 'qua': 'quá', 'met': 'mệt',
        'moi': 'mỏi', 'cam': 'cảm', 'on': 'ơn',
        'nhieu': 'nhiều', 'hay': 'hay', 'lam': 'lắm',
        'mon': 'món', 'an': 'ăn', 'thoi': 'thời',
        'tiet': 'tiết', 'binh': 'bình', 'thuong': 'thường',
        'cong': 'công', 'viec': 'việc', 'dinh': 'định',
        'phim': 'phim', 'vi': 'vì', 'that': 'thất',
        'bai': 'bại', 'ngay': 'ngày', 'mai': 'mai',
        'di': 'đi', 'hoc': 'học', 'duoc': 'được',
        'khong': 'không', 'tot': 'tốt', 'dep': 'đẹp',
        'xau': 'xấu', 'ban': 'bạn',
        'yeu': 'yêu', 'ghet': 'ghét', 'thich': 'thích',
        'chan': 'chán', 'so': 'sợ', 'lo': 'lo',
    }

    tokens = re.findall(r"[\wáàảãạăắằẳẵặâấầẩẫậđéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ]+|[^\w\s]", text)

    result = []
    for token in tokens:
        low = token.lower()
        if low in replacements:
            replaced = replacements[low]
            if token[0].isupper():
                replaced = replaced.capitalize()
            result.append(replaced)
        else:
            result.append(token)

    return ''.join([
        (' ' + w if re.match(r'\w', w) and i != 0 and re.match(r'\w', result[i - 1]) else w)
        for i, w in enumerate(result)
    ])

def main():
    st.title('Trợ lý phân loại cảm xúc tiếng Việt')

    connection = get_db_connection()

    with st.form('input_form'):
        input_text = st.text_input('Nhập', '')
        is_submitted = st.form_submit_button('Phân loại cảm xúc')

    if is_submitted:
        if not input_text or len(input_text.strip()) < 5:
            st.error('Câu quá ngắn. Vui lòng nhập ít nhất 5 ký tự.')
            st.toast('Câu quá ngắn', icon='❌')
        else:
            try:
                with st.spinner('Đang phân loại...'):
                    classifier = get_sentiment_classifier()
                    label, score, original_label, json_output = analyze_sentiment(classifier, input_text)
                    save_sentiment_record(connection, input_text, label)
                
                color = get_sentiment_color(label)
                st.markdown(
                    f'<div style="padding: 10px; border-radius: 5px; border-left: 4px solid {color};">'
                    f'<strong>Kết quả:</strong> <span style="color: {color}; font-weight: bold; font-size: 1.2em;">{label}</span> '
                    f'<span style="color: #666;">(Độ tin cậy: {score:.2f})</span></div>',
                    unsafe_allow_html=True
                )
                st.caption(f'Nhãn gốc từ model: {original_label}')
                
                with st.expander('Output'):
                    st.json(json_output)
            except Exception as e:
                st.error(f'Lỗi: {e}')

    st.subheader('Danh sách lịch sử phân loại')
    history_records = get_sentiment_history(connection, limit=50)
    if history_records:
        df_history = pd.DataFrame(history_records, columns=['Câu văn', 'Cảm xúc', 'Thời gian'])
        df_history['Cảm xúc'] = df_history['Cảm xúc'].apply(format_sentiment_html)
        st.markdown(
            df_history.to_html(escape=False, index=False),
            unsafe_allow_html=True
        )
    else:
        st.info('Chưa có bản ghi nào.')

if __name__ == '__main__':
    main()