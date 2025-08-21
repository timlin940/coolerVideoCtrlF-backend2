import os
import psycopg2
from datetime import timedelta
from sentence_transformers import SentenceTransformer
#from app.chroma_client import ChromaDBClient
from chromadb import PersistentClient
from chromadb.config import Settings
import json
import google.generativeai as genai
import re


api_key =  os.getenv("GEMINI_API_KEY")

genai.configure(api_key=api_key)
# === 共用工具 ===
def parse_time(ts):
    h, m, s = ts.split(":")
    return timedelta(hours=int(h), minutes=int(m), seconds=float(s))

def find_start_index(transcript_json, target_time):
    """從字幕中找到第一個開始時間 >= target_time 的 index"""
    for i, seg in enumerate(transcript_json):
        if parse_time(seg["start"]) >= parse_time(target_time):
            return i
    return len(transcript_json)

def slice_transcript_block(transcript_json, start_index, max_tokens=2000):
    total_tokens = 0
    block = []
    for i in range(start_index, len(transcript_json)):
        text = transcript_json[i]["content"]
        approx_tokens = len(text) // 4
        if total_tokens + approx_tokens > max_tokens:
            break
        block.append(transcript_json[i])
        total_tokens += approx_tokens
    return block

def gemini_segment_and_summarize(subtitle_slices):
    text_block = ""
    for seg in subtitle_slices:
        text_block += f"[{seg['start']} - {seg['end']}] {seg['content']}\n"

    prompt = f"""
你是語意分段助手，請依據下方字幕（包含時間與對應文字）進行切段摘要：
{text_block}

請根據語意切分這些字幕段落，每段需要：
1. 開始時間與結束時間（不能超出提供資料）
2. 每段摘要（1～2 句**英文**）
3. 每30秒左右切分一次(必須小於等於30秒)，如果有重複的內容可以合併成一段。

請回傳**純 JSON 陣列**，不要加上```json或任何Markdown符號。
格式如下：
[
  {{"start": "00:00:00", "end": "00:00:05", "summary": "段落摘要"}}
]
"""
    model_ai = genai.GenerativeModel('gemini-1.5-flash')
    response = model_ai.generate_content(prompt)
    raw_text = response.text if hasattr(response, "text") else response.parts[0].text
    print("📥 Gemini 回傳原文：\n", raw_text)

    # 用 regex 提取第一個合法的 JSON 陣列
    json_match = re.search(r"\[\s*{.*?}\s*\]", raw_text, re.DOTALL)
    if not json_match:
        print("❌ 無法從 Gemini 回傳中提取合法 JSON。")
        print(raw_text)
        return []

    try:
        return json.loads(json_match.group(0))
    except json.JSONDecodeError as e:
        print("❌ JSON 格式錯誤：", e)
        print("👉 錯誤輸入：\n", json_match.group(0))
        return []

def process_subtitles_with_gemini(video_id, transcript_json, url, title, model_st, collection_chunk):
    current_time = transcript_json[0]["start"]

    while True:
        start_idx = find_start_index(transcript_json, current_time)
        if start_idx >= len(transcript_json):
            break

        block = slice_transcript_block(transcript_json, start_idx)
        if not block:
            break

        try:
            segments = gemini_segment_and_summarize(block)
            if not segments:
                break

            for i, seg in enumerate(segments):
                summary = seg["summary"].strip()
                chunk_id = f"{video_id}_gemini_{seg['start']}_{seg['end']}"
                
                if chunk_id in existing_ids_chunk:
                    continue  # 已存在就跳過該段

                vector = model_st.encode([summary])[0].tolist()
                collection_chunk.add(
                    documents=[summary],
                    embeddings=[vector],
                    ids=[chunk_id],
                    metadatas=[{
                        "video_id": video_id,
                        "start": seg["start"],
                        "end": seg["end"],
                        "summary": summary,
                        "url": url,
                        "title": title
                    }]
                )
                print(f"✅ 儲存段落：{chunk_id}")

            # ✅ 儲存完所有段落後，再檢查時間有無前進
            next_time = segments[-1]["end"]
            if current_time == next_time:
                print(f"⚠️ 偵測到無時間前進，跳出循環：{current_time}")
                break
            current_time = next_time

        except Exception as e:
            print(f"❌ Gemini 分析錯誤：video_id={video_id}，錯誤：{e}")
            break

# === ChromaDB 初始化 ===
# client = ChromaDBClient.get_instance().get_client() #原版
client = PersistentClient(path="test_chroma_db", settings=Settings(allow_reset=True))
collection_tt = client.get_or_create_collection(name="title_tag_emb")
collection_st = client.get_or_create_collection(name="summary_transcription_emb")
collection_chunk = client.get_or_create_collection(name="transcription_chunks_emb")

# === 模型初始化 ===
model_tt = SentenceTransformer("paraphrase-MiniLM-L6-v2")
model_st = SentenceTransformer("BAAI/bge-m3")

# === PostgreSQL 連線 ===
print("🔐 連線到 PostgreSQL 中...")
#DATABASE_URL = os.getenv("DATABASE_URL") or "postgresql://postgres:pMHQKXAVRWXxhylnCiKOmslOKgVbjdvM@switchyard.proxy.rlwy.net:43353/railway"
DATABASE_URL = "postgresql://postgres:Functrol@localhost:5432/postgres"
conn = psycopg2.connect(DATABASE_URL)
cursor = conn.cursor()

# === 抓影片基本資料 + 字幕 ===
cursor.execute("""
    SELECT 
        v.id AS video_id,
        v.url,
        v.title,
        v.summary,
        v.transcription,
        v.transcription_with_time,
        -- 聚合 tag 名稱
        (
            SELECT STRING_AGG(t.name, '; ')
            FROM jsonb_array_elements_text(v.tag_ids) AS tag_id_text
            JOIN tags t ON t.id = tag_id_text::int
        ) AS tags
    FROM video_categories vc 
    JOIN videos v ON v.id = vc.video_id 
    JOIN categories c ON vc.category_id = c.id
    WHERE v.transcription_with_time IS NOT NULL AND v.id > 455 AND v.id < 461
    GROUP BY v.id, v.url, v.title, v.summary, v.transcription, v.transcription_with_time, v.tag_ids
    ORDER BY v.id
""")                                                # 可以透過兩個and那行來專門存特定範圍的片段
rows = cursor.fetchall()
columns = [desc[0] for desc in cursor.description]

# === 已存在的 ID（避免重複） ===
existing_ids_tt = set(collection_tt.get()["ids"])
existing_ids_st = set(collection_st.get()["ids"])
existing_ids_chunk = set(collection_chunk.get()["ids"])

# === 欄位對應模型與 collection ===
field_mapping = {
    "title": (collection_tt, model_tt),
    "tags": (collection_tt, model_tt),
    "summary": (collection_st, model_st),
    "transcription": (collection_st, model_st),
}

for row in rows:
    row_dict = dict(zip(columns, row))
    video_id = str(row_dict["video_id"])
    url = row_dict["url"]
    title = row_dict["title"]
    t_with_time = row_dict["transcription_with_time"]

    # === 儲存 title/tags/summary/transcription 向量 ===
    for field, (collection, model) in field_mapping.items():
        content = (row_dict.get(field) or "").strip()
        if not content:
            continue

        uid = f"{video_id}_{field}"
        if (field in ["title", "tags"] and uid in existing_ids_tt) or \
           (field in ["summary", "transcription"] and uid in existing_ids_st):
            print(f"⚠️ 已存在：{uid}，跳過儲存")
            continue

        vector = model.encode([content])[0].tolist()
        collection.add(
            documents=[content],
            embeddings=[vector],
            ids=[uid],
            metadatas=[{
                "video_id": video_id,
                "field": field,
                "url": url,
                "title": title,
                "tags": row_dict["tags"]
            }]
        )
    # === 處理 Gemini 字幕分段摘要（加在每部影片迴圈內）===
    try:
        process_subtitles_with_gemini(
            video_id=video_id,
            transcript_json=t_with_time,
            url=url,
            title=title,
            model_st=model_st,
            collection_chunk=collection_chunk
        )
    except Exception as e:
        print(f"❌ 處理 Gemini 分段失敗：video_id={video_id}，錯誤：{e}")

cursor.close()
conn.close()
print("✅ 已成功將所有影片欄位與字幕片段向量儲存完成！")