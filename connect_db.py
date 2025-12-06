import sqlite3

DB_PATH = 'sentiments.db'

def init_db():
    """Khởi tạo database."""
    connection = sqlite3.connect(DB_PATH)
    cursor = connection.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS sentiments
                   (id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT, sentiment TEXT, timestamp TEXT)''')
    connection.commit()
    connection.close()
    print('Đã khởi tạo', DB_PATH)

if __name__ == '__main__':
    init_db()
