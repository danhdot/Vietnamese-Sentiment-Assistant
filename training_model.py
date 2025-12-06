import sqlite3
import pandas as pd
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)
from datasets import Dataset
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import os
from datetime import datetime, timezone

DB_PATH = 'sentiments.db'
MODEL = 'wonrax/phobert-base-vietnamese-sentiment'
OUTPUT_DIR = './tuned_model'

def load_training_data():
    conn = sqlite3.connect(DB_PATH)
    query = "SELECT text, sentiment FROM sentiments"
    df = pd.read_sql_query(query, conn)
    conn.close()
    print(f"Đã load {len(df)} cases")
    
    # Ánh xạ nhãn cảm xúc sang số
    label = {
        'TÍCH CỰC': 2,
        'RẤT TÍCH CỰC': 2,
        'TRUNG LẬP': 1,
        'TIÊU CỰC': 0,
        'RẤT TIÊU CỰC': 0,
    }
    
    df['label'] = df['sentiment'].map(label)
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)
    
    print("Kết quả nhãn như sau:")
    print(df['sentiment'].value_counts())
    
    return df

def prepare_dataset(df):
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

    # Tạo datasets
    train_dataset = Dataset.from_pandas(train_df[['text', 'label']])
    test_dataset = Dataset.from_pandas(test_df[['text', 'label']])
    
    return train_dataset, test_dataset

def tokenize_function(examples, tokenizer):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    
    return {
        'accuracy': accuracy,
        'f1': f1
    }

def add_manual_training_data(conn):
    cur = conn.cursor()
    
    # Dữ liệu
    training_samples = [
        # Tiêu cực (Negative)
        ("Không vui chút nào", "TIÊU CỰC"),
        ("Tôi bực mình quá", "TIÊU CỰC"),
        ("Món ăn này dở quá", "TIÊU CỰC"),
        ("Nản hết sức", "TIÊU CỰC"),
        ("Tôi buồn vì thất bại", "TIÊU CỰC"),
        ("Thật tệ hại", "TIÊU CỰC"),
        ("Không thích cái này", "TIÊU CỰC"),
        ("Quá tệ", "TIÊU CỰC"),
        ("Thất vọng quá", "TIÊU CỰC"),
        ("Quá phiền", "TIÊU CỰC"),
        ("Không ổn chút nào", "TIÊU CỰC"),
        ("Tôi sợ quá", "TIÊU CỰC"),
        ("Tôi bị ngu", "TIÊU CỰC"),
        
        # Tích cực (Positive)
        ("Rất tuyệt", "TÍCH CỰC"),
        ("Tôi rất vui", "TÍCH CỰC"),
        ("Tôi thích cái này", "TÍCH CỰC"),
        ("Hoàn toàn hài lòng", "TÍCH CỰC"),
        ("Hợp lý ghê", "TÍCH CỰC"),
        ("Tôi yêu điều này", "TÍCH CỰC"),
        ("Quá đỉnh", "TÍCH CỰC"),
        ("Tôi thấy rất hài lòng", "TÍCH CỰC"),
        ("Quá ổn", "TÍCH CỰC"),
        ("Tôi đánh giá cao điều này", "TÍCH CỰC"),
        ("Thật tuyệt", "TÍCH CỰC"),
        ("Tốt lắm", "TÍCH CỰC"),
        ("Hoàn hảo", "TÍCH CỰC"),
        
        # Trung lập (Neutral)
        ("Tôi đang suy nghĩ", "TRUNG LẬP"),
        ("Tôi là Danh Đạt", "TRUNG LẬP"),
        ("Chưa biết nữa", "TRUNG LẬP"),
        ("Đây là đâu", "TRUNG LẬP"),
        ("Bình thường", "TRUNG LẬP"),
        ("Tôi chưa quyết định", "TRUNG LẬP"),
        ("Bây giờ là 7 giờ sáng", "TRUNG LẬP"),
    ]
    
    from datetime import datetime
    for text, sentiment in training_samples:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        cur.execute('INSERT INTO sentiments (text, sentiment, timestamp) VALUES (?, ?, ?)', 
                   (text, sentiment, ts))
    
    conn.commit()
    print(f"Đã thêm {len(training_samples)} cases training vào database")

def train_model():
    print("START TRAINING MODEL SENTIMENT ANALYSIS")
    
    if not os.path.exists(DB_PATH):
        print("Hãy run streamlit run app.py do không có database.")
        return
    
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM sentiments")
    count = cur.fetchone()[0]
    
    if count < 20:
        print(f"Đang có {count} cases. Đang thêm dữ liệu")
        add_manual_training_data(conn)
    
    conn.close()

    df = load_training_data()
    
    if len(df) < 10:
        print("Cần ít nhất 10 cases.")
        return
    
    # Chuẩn bị datasets
    print("\nChuẩn bị datasets")
    train_dataset, test_dataset = prepare_dataset(df)
    
    # Load tokenizer và model
    print(f"\nSử dụng model: {MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL,
        num_labels=3,  # 3 labels: tiêu cực (0), trung lập (1), tích cực (2)
        ignore_mismatched_sizes=True
    )
    
    # Tokenize datasets
    print("\nTokenizing...")
    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer), 
        batched=True
    )
    test_dataset = test_dataset.map(
        lambda x: tokenize_function(x, tokenizer), 
        batched=True
    )
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
        logging_strategy="steps",
        logging_steps=50,
        warmup_ratio=0.1
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Train
    print("\nStart training.")
    trainer.train()
    
    # Đánh giá
    print("\nĐánh giá model:")
    results = trainer.evaluate()
    print("\nKết quả:")
    for key, value in results.items():
        print(f"  {key}: {value:.4f}")
    
    # Lưu model
    print(f"\n Đã lưu model vào thư mục {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    train_model()
