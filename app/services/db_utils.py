import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()  # 載入 .env

def login_postgresql():
    print("🔐 正在登入本地 PostgreSQL 資料庫...")

    DATABASE_URL = os.getenv("DATABASE_URL")
    if not DATABASE_URL:
        print("❌ 找不到 DATABASE_URL，請檢查 .env 設定")
        exit()

    try:
        conn = psycopg2.connect(DATABASE_URL)
        print("✅ 成功連線到 PostgreSQL！")
        return conn
    except Exception as e:
        print("❌ 連線失敗：", e)
        exit()

# Test the connection
# if __name__ == "__main__":
#     login_postgresql()