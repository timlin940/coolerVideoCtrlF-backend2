import subprocess
import json, re, os
import psycopg2
from datetime import datetime
import google.generativeai as genai
import time
import random
from dotenv import load_dotenv
load_dotenv()

def generate_highlights_text_with_gemini(subtitles) :
    prompt = f"""
        你是教學助教。請閱讀以下逐段字幕內容，萃取最重要的5個重點，。
        **輸出規則**：
        - 用繁體中文，可以保留英文專有名詞
        - 每個重點 ≤ 40 字
        - 重點要清楚、明確
        - 每個重點獨立一行，且**以數字作為開頭**
        - **不要**輸出任何標題、說明、JSON、程式碼框或多餘文字
        - 範本，**重點之間不需要空一行**：
        - **1.這是第一個重點**
        - **2.這是第二個重點**

        字幕：
        {subtitles}
        """
    try:
        model = genai.GenerativeModel("gemini-2.5-flash-lite")
        resp = model.generate_content(prompt)
        resp = resp.text.strip()
        return resp
    except Exception as e:
        print("❌ Gemini 產生重點失敗：", e)
        return ""

if __name__ == "__main__":
    api_key =  os.getenv("GEMINI_API_KEY")
    genai.configure(api_key=api_key)
    DATABASE_URL = (
            os.getenv("DATABASE_URL") or 
            "postgresql://postgres:pMHQKXAVRWXxhylnCiKOmslOKgVbjdvM@switchyard.proxy.rlwy.net:43353/railway"
        )
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id,  transcription
        FROM videos
        where highlights IS NULL OR highlights = ''
        ORDER BY id
    """)
    videos = cursor.fetchall()
    print(f"🔍 共 {len(videos)} 支影片待處理")

    for video in videos:
        video_id, transcription= video
        print(f"▶️ 處理影片 {video_id}")

        highlights = generate_highlights_text_with_gemini(transcription)
    
        if not highlights.strip():
            print(f"❌ 影片 {video_id} 的重點內容為空，跳過")
            continue

        # 更新資料庫
        try:
            cursor.execute("""
                UPDATE videos
                SET highlights = %s
                WHERE id = %s
            """, (highlights,  video_id))
            conn.commit()
            print(highlights)
            print(f"✅ 影片 {video_id} 的重點已更新")
        except Exception as e:
            conn.rollback()
            print(f"❌ 更新影片 {video_id} 的重點失敗：", e)

        # 避免觸發速率限制，隨機等待 1～3 秒
        time.sleep(random.uniform(1, 3))

    cursor.close()
    conn.close()