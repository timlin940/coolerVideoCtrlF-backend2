import subprocess
import json, re, os
import psycopg2
from datetime import datetime
import google.generativeai as genai
import time
import random
from dotenv import load_dotenv
load_dotenv()

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

#我把yt-dlp.exe放在backend資料夾下，用來搜尋YouTube影片

# ✅ 呼叫
api_key =  os.getenv("GEMINI_API_KEY")

genai.configure(api_key="AIzaSyAkbe3eQXu4VLJZ8oWlo3RYjqVDy1h4JKQ")

def time_str_to_str(time_str):
    parts = time_str.split(":")
    if len(parts) == 3:
        h = int(parts[0])
        m = int(parts[1])
        s = int(float(parts[2]))
        return f"{h}:{m:02}:{s:02}"
    elif len(parts) == 2:
        m = int(parts[0])
        s = int(float(parts[1]))
        return f"0:{m:02}:{s:02}"
    else:
        return "0:00:00"

def seconds_to_time_str(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02}:{m:02}:{s:02}"

def login_postgresql():
    print(" 請登入 PostgreSQL 資料庫")
    
    try:
        DATABASE_URL = (
            os.getenv("DATABASE_URL") or 
            "postgresql://postgres:pMHQKXAVRWXxhylnCiKOmslOKgVbjdvM@switchyard.proxy.rlwy.net:43353/railway"
        )
        conn = psycopg2.connect(DATABASE_URL)
        print(" 成功連線到 PostgreSQL！")
        return conn
    except Exception as e:
        print(" 連線失敗：", e)
        exit()

def search_youtube_with_subtitles(keyword, max_results=10):
    yt_dlp_path = r"C:\Users\user\Desktop\Functrol\backend\yt-dlp.exe"
    print(f"\U0001f50d 搜尋關鍵字：{keyword}")
    command = [
        yt_dlp_path,
        f"ytsearch{max_results}:{keyword}",
        "--dump-json",
        "--no-warnings"
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        lines = result.stdout.strip().split('\n')
        valid_videos = []
        for line in lines:
            video_data = json.loads(line)
            if video_data.get("subtitles") or video_data.get("automatic_captions"):
                valid_videos.append({
                    "title": video_data.get("title"),
                    "url": video_data.get("webpage_url"),
                    "description": video_data.get("description"),
                    "duration": video_data.get("duration_string"),
                    "channel": video_data.get("channel")
                })
        return valid_videos
    except subprocess.CalledProcessError as e:
        print(" 執行 yt-dlp 發生錯誤：", e)
        return []
    
def generate_summary_with_gemini(text, prompt="請為以下影片字幕生成一段精簡的英文摘要(避免逐句翻譯)："):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt + text)
        return response.text.strip()
    except Exception as e:
        print("❌ Gemini 摘要失敗：", e)
        return False
    
def predict_topic_with_gemini(summary_text):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = (
            f"這是一段YouTube影片的摘要內容：{summary_text}\n"
            "請根據以下主題分類中，選出最適合的2個主題（只回傳英文主題名稱，用逗號分隔）：\n"
            "Computer Science, Law, Mathematics, Physics, Chemistry, Biology, Earth Science, History, Geography, Sports, Astronomy, Daily Life。\n"
            "請勿自行創造其他分類。"
        )
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print("❌ Gemini 主題分類失敗：", e)
        return "Daily Life"  # fallback 預設值

def get_or_create_tags(summary, conn):
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT id, name FROM tags")
        tag_rows = cursor.fetchall()
        tag_dict = {t.lower(): i for i, t in tag_rows}
        tag_list = list(tag_dict.keys())

        prompt = f"""
    你是專業的知識分類專家，請根據下列影片摘要，嚴格遵守規則選出最適合的 3 個英文標籤（用逗號分隔，不得多於 3 個）：

    摘要內容如下：
    「{summary}」

    已知標籤清單如下：
    {', '.join(tag_list)}

    請**務必遵守以下規則**：

    1. 優先從清單中挑選 3 個最貼近摘要主題的標籤。
    2. 若清單中沒有合適的，請**直接創造新的標籤（最多 3 個）**。
    3. **禁止創造與現有標籤語意重複或相近的標籤**（例如已有 Machine Learning，就不能創造 ML 或 AI ML）。
    4. 新標籤必須是**清楚、簡短、意義明確的英文詞彙**，且**限制為兩個單字以內**。
    5. 僅回傳標籤內容，用英文逗號分隔，**不要附加任何文字、引號、格式符號或說明**。

    範例正確回覆格式（僅供參考）：
    AI, Blockchain, Cybersecurity
    """
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        tags_raw = response.text.strip()
        tags = [t.strip() for t in tags_raw.split(",") if t.strip()]

        tag_ids = []
        for tag in tags:
            tag_lower = tag.lower()
            if tag_lower in tag_dict:
                tag_ids.append(tag_dict[tag_lower])
            else:
                cursor.execute("INSERT INTO tags (name) VALUES (%s) RETURNING id", (tag,))
                new_id = cursor.fetchone()[0]
                tag_ids.append(new_id)
                conn.commit()

        return tag_ids

    except Exception as e:
        print("❌ 產生標籤失敗：", e)
        return []

def download_and_save_to_postgresql(video_url, title, description, conn, language="en"):
    print(f"\U0001f3ac 處理影片：{video_url}")
    video_id = video_url.split("v=")[-1]

    cursor = conn.cursor()
    cursor.execute("SELECT id FROM videos WHERE url = %s", (video_url,))
    if cursor.fetchone():
        print(f" 影片已存在於資料庫中，略過：{video_url}")
        return
    try:
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            try:
                # 優先使用手動字幕
                transcript = transcript_list.find_manually_created_transcript([language]).fetch()
            except NoTranscriptFound:
                # 若沒有手動字幕，再找自動字幕
                transcript = transcript_list.find_generated_transcript([language]).fetch()
        except (TranscriptsDisabled, NoTranscriptFound):
            print(f"❌ 此影片無 {language} 字幕：{video_url}")
            return
        structured_subtitles = []
        output_lines = []

        for i, entry in enumerate(transcript):
            start = entry.start
            duration = entry.duration
            content = entry.text.strip()

            # 格式化為 mm:ss
            mmss = time_str_to_str(seconds_to_time_str(start))

            # 移除 HTML 標籤與雜訊
            content = re.sub(r"<.*?>", "", content)
            content = re.sub(r"\[.*?\]", "", content)
            content = re.sub(r"\s+", " ", content)

            # 忽略空段落
            if not content:
                continue

            # 過濾重複
            if structured_subtitles and content in structured_subtitles[-1]["content"]:
                continue

            start_sec = start
            end_sec = start + duration

            structured_subtitles.append({
                "start": time_str_to_str(seconds_to_time_str(start_sec)),
                "end": time_str_to_str(seconds_to_time_str(end_sec)),
                "content": content
            })
            output_lines.append(content)

        subtitle_text = "\n".join(output_lines)

        # 儲存影片長度（取最後字幕結束時間）
        if transcript:
            last_end = transcript[-1].start + transcript[-1].duration
            duration_str = time_str_to_str(seconds_to_time_str(last_end))
            duration_sec = int(last_end)
            if duration_sec < 180:
                print(f" 影片長度僅 {duration_str}，少於 3 分鐘，略過儲存")
                return
            if duration_sec > 7200:
                print(f" 影片長度 {duration_str}，大於 2 小時，略過儲存")
                return
        else:
            duration_str = ""

        # 清理字幕內容
        subtitle_text = clean_text(subtitle_text)

        # 抽出內嵌網址
        embed_url = f"https://www.youtube.com/embed/{video_id}"

        # 做 summary + 主題分類
        summary = generate_summary_with_gemini(subtitle_text)
        assigned_categories = predict_topic_with_gemini(summary)

        tag_ids = get_or_create_tags(summary, conn) 

        assigned_categories = [t.strip() for t in assigned_categories.split(",")]
        if summary is False:
            print(f"❌ 摘要生成失敗，略過儲存：{video_url}")
            return
        # 儲存到資料庫
        cursor.execute("""
            INSERT INTO videos (url, title, description, summary, transcription, transcription_with_time, duration_str, embed_url, created_at,tag_ids)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s,%s)
            RETURNING id;
        """, (
            video_url, title, description, summary, subtitle_text,
            json.dumps(structured_subtitles, ensure_ascii=False),
            duration_str, embed_url, datetime.utcnow(),json.dumps(tag_ids)
        ))

        new_video_id = cursor.fetchone()[0]
        for topic in assigned_categories:
            cursor.execute("SELECT id FROM categories WHERE topic = %s", (topic,))
            result = cursor.fetchone()
            if result:
                category_id = result[0]
                cursor.execute("INSERT INTO video_categories (video_id, category_id) VALUES (%s, %s)", (new_video_id, category_id))
            else:
                print(f"⚠️ 找不到分類：{topic}，略過此分類")

        conn.commit()
        print(f"✅ 成功儲存影片：{title}，主題：{', '.join(assigned_categories)}")

    except Exception as e:
        print("❌ 發生錯誤：", e)

def clean_text(text):#清理字幕檔
    text = re.sub(r'WEBVTT.*?\n', '', text, flags=re.DOTALL)
    text = re.sub(r'\[[^\]]+\]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text

if __name__ == "__main__":#重要事項:不要一次輸入太多關鍵字，否則會造成API呼叫過快而被封鎖
    keyword = input("請輸入關鍵字（多個關鍵字用逗號分隔）：").split(",")
    keyword = [k.strip() for k in keyword if k.strip()]  

    conn = login_postgresql()

    for key in keyword:
        videos = search_youtube_with_subtitles(key, max_results=10 )
        for i, video in enumerate(videos, 1):
            time.sleep(60 + random.randint(0, 5))
            print(f"{i}. {video['title']}")
            print(f"連結: {video['url']}")
            print(f"頻道: {video['channel']}")
            print(f"時長: {video['duration']}")
            download_and_save_to_postgresql(video['url'], video['title'], video.get('description', ''), conn)
        time.sleep(200 + random.randint(0, 5))  # 避免過快呼叫API
    conn.close()