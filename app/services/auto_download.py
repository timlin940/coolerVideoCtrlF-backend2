# 在背景自動下載影片
import subprocess
import json, re, os
import psycopg2
from datetime import datetime
import google.generativeai as genai
import time
import random
import requests
from dotenv import load_dotenv
from app.services.vectordb_search_for_main import get_latest_id,store_emb
# 自動下載

load_dotenv()

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

#我把yt-dlp.exe放在backend資料夾下，用來搜尋YouTube影片

# ✅ 呼叫
api_key =  os.getenv("GEMINI_API_KEY")

genai.configure(api_key=api_key)

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
        model = genai.GenerativeModel("gemini-2.5-flash-lite")
        response = model.generate_content(prompt + text)
        return response.text.strip()
    except Exception as e:
        print("❌ Gemini 摘要失敗：", e)
        return False
    
def predict_topic_with_gemini(summary_text):
    try:
        model = genai.GenerativeModel("gemini-2.5-flash-lite")
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
        model = genai.GenerativeModel("gemini-2.5-flash-lite")
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

def is_embeddable(video_url: str, timeout=5) -> bool:
    """
    檢查 YouTube 影片是否允許嵌入 (oEmbed API)
    如果不能嵌入，代表影片可能是私有、需要登入、地區限制或被下架。
    """
    try:
        r = requests.get(
            "https://www.youtube.com/oembed",
            params={"url": video_url, "format": "json"},
            timeout=timeout,
        )
        return r.status_code == 200
    except requests.RequestException:
        return False

def download_and_save_to_postgresql(video_url, title, description, conn, language="en"):
    print(f"\U0001f3ac 處理影片：{video_url}")
    # ✅ 嵌入檢查：避免存到不可觀看的影片
    if not is_embeddable(video_url):
        print(f"❌ 此影片無法嵌入或不可觀看，略過：{video_url}")
        return
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

        # 取得或創建標籤
        tag_ids = get_or_create_tags(summary, conn) 
        # 生成重點helights
        highlights = generate_highlights_text_with_gemini(subtitle_text)


        assigned_categories = [t.strip() for t in assigned_categories.split(",")]
        if summary is False:
            print(f"❌ 摘要生成失敗，略過儲存：{video_url}")
            return
        # 儲存到資料庫
        cursor.execute("""
            INSERT INTO videos (url, title, description, summary, transcription, transcription_with_time, duration_str, embed_url, created_at,tag_ids, highlights)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s,%s,%s)
            RETURNING id;
        """, (
            video_url, title, description, summary, subtitle_text,
            json.dumps(structured_subtitles, ensure_ascii=False),
            duration_str, embed_url, datetime.utcnow(),json.dumps(tag_ids), highlights
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

#  主程式：自動下載影片並存到資料庫，還需要store_all_embd
def auto_download(keywords):
    conn = login_postgresql()
    for keyword in keywords:
        videos = search_youtube_with_subtitles(keyword, max_results=2)
        print(f"🔍 找到 {len(videos)} 支有字幕的影片")
        for video in videos:
            download_and_save_to_postgresql(video["url"], video["title"], video["description"], conn, language="en")
            time.sleep(20 + random.randint(0, 5))
        # 用local_view_collection找現在最新的collection到哪裡
        #########################################################################################(以上下載影片到SQL沒問題)
        latest_id = get_latest_id()#出現vocab問題
        # 用store_all_embd儲存向量
        print("最新的資料庫:",latest_id)
        store_emb(latest_id, conn)
    print("影片下載並處理完成")
    conn.close()