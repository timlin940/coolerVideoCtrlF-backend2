import re
from chromadb import PersistentClient
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import torch
from app.services.llm_expand import generate_related_queries  # 確保你引入 LLM 擴展函數
import psycopg2
import os
from datetime import timedelta
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv
load_dotenv()
from app.chroma_client import ChromaDBClient
import google.generativeai as genai
import json  # 確保已匯入


api_key =  os.getenv("GEMINI_API_KEY")

genai.configure(api_key=api_key)

# === 初始化 ChromaDB client & collection ===
client = ChromaDBClient.get_instance().get_client()
collection_tt = client.get_or_create_collection(name="title_tag_emb")
collection_st = client.get_or_create_collection(name="summary_transcription_emb")
collection_chunks = client.get_or_create_collection(name="transcription_chunks_emb")

# 取得所有 collections
collections = client.list_collections()

# === 模型 ===
model_tt = SentenceTransformer("paraphrase-MiniLM-L6-v2", device='cuda' if torch.cuda.is_available() else 'cpu')
model_st = SentenceTransformer("BAAI/bge-m3", device='cuda' if torch.cuda.is_available() else 'cpu')
model = genai.GenerativeModel("gemini-2.5-flash-lite")

# === 公用 cosine similarity 函數 ===
def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    return dot / (norm_a * norm_b + 1e-9)

# === 查詢字幕片段最相關的起始時間（取前一段） === ###最後return的best_idx有-1才是前一段
def get_best_chunk_start(video_id: str, query: str):
    query_emb = model_st.encode([query])[0]

    # 使用 collection.query 查詢最相關的前 20 個片段
    res = collection_chunks.query(
        query_embeddings=[query_emb],
        n_results=20,
        where={"video_id": video_id},
        include=["metadatas", "distances"]  # 包含距離分數
    )

    # Step 4: 整理 query 結果準備給 Gemini 重排序
    # query() 方法返回的是巢狀列表結構 [[...]]
    metadatas = res["metadatas"][0]  # 取第一個查詢的結果
    
    # 準備給 Gemini 重排序的資料
    segments_for_rerank = []
    for i, meta in enumerate(metadatas):
        segment_data = {
            "start": meta.get("start", 0),
            "summary": meta.get("summary", ""),
        }
        segments_for_rerank.append(segment_data)
    
    # 準備給 Gemini 的 prompt
    segments_text = ""
    for i, segment in enumerate(segments_for_rerank):
        segments_text += f"片段 {i+1}:\n"
        segments_text += f"時間: {segment.get('start', 'N/A')}秒\n"
        segments_text += f"內容摘要: {segment.get('summary', '')}\n"
        segments_text += f"---\n"
    
    prompt = f"""
請根據使用者查詢對以下影片片段進行相關性排序。

使用者查詢: "{query}"

影片片段:
{segments_text}

請仔細分析每個片段與查詢的相關性，考慮以下因素：
1. 內容主題的匹配度
2. 關鍵詞的相關性
3. 語義相似性
4. 實用性和價值

請按照相關性從高到低排序，並以 JSON 格式回傳結果。格式如下：
{{
    "ranking": [1, 3, 2, 5, 4, ...],
    "explanation": "簡要說明排序的理由"
}}

其中 ranking 數組包含重新排序後的片段編號（1-based index）。
"""
    
    try:
        response = model.generate_content(prompt)
        result_text = response.text.strip()
        
        # 嘗試解析 JSON 回應
        if result_text.startswith("```json"):
            result_text = result_text[7:-3].strip()
        elif result_text.startswith("```"):
            result_text = result_text[3:-3].strip()
        
        result = json.loads(result_text)
        ranking = result.get("ranking", list(range(1, len(segments_for_rerank) + 1)))
        
        # 取出第一名片段的 start 時間
        if ranking:
            top_segment_idx = ranking[0] - 1  # 1-based to 0-based index
            best_segment = segments_for_rerank[top_segment_idx]
            return best_segment["start"]

    except Exception as e:
        print(f"Gemini 重排序失敗: {e}")
        # 如果 Gemini 失敗，返回原始順序的第一個片段的 start
        return segments_for_rerank[0]["start"] if segments_for_rerank else None



# === 主流程 ===
def search_videos_with_vectorDB(query: str, k=5):
    expanded_queries = generate_related_queries(query)
    weights = [5, 4, 3, 2, 2, 2]
    final_scores = {}
    need_download = False # 預設不需要下載
    for i, query_text in enumerate(expanded_queries):
        video_scores = {}
        weight = weights[i]

        q_tt_emb = model_tt.encode([query_text])[0]
        q_st_emb = model_st.encode([query_text])[0]

        # Title 專屬前 10
        results_title = collection_tt.query(
            query_embeddings=[q_tt_emb],
            n_results=10,
            where={"field": "title"},
            include=["metadatas", "embeddings"]
        )

        # 處理 summary
        results_st = collection_st.query(
            query_embeddings=[q_st_emb],
            n_results=10,
            include=["metadatas", "embeddings"]
        )

        # 建立候選影片清單（僅 title 命中的影片）
        candidate_vids = set()
        for emb, meta in zip(results_title["embeddings"][0], results_title["metadatas"][0]):
            vid = meta["video_id"]
            score = (cosine_similarity(q_tt_emb, emb) + 1.0)/2.0
            if score < 0.6:
                score = 0  # 過濾掉低於 0.6 的相似度
            video_scores.setdefault(vid, {"title": 0, "tags": 0, "summary": 0})
            video_scores[vid]["title"] = score
            candidate_vids.add(vid)

        # 只用一個 where 條件：video_id in 候選
        tags_docs = collection_tt.get(
            where={"video_id": {"$in": list(candidate_vids)}},  # 只放「一個」operator
            include=["metadatas", "embeddings"]
        )

        # 對每支影片取「所有 tag 向量」的最大相似度當作該影片 tags 分數
        for emb, meta in zip(tags_docs.get("embeddings", []), tags_docs.get("metadatas", [])):
            if meta.get("field") != "tags":
                continue  # 只算 tags
            vid = meta.get("video_id")
            if not vid:
                continue
            score = (cosine_similarity(q_tt_emb, emb) + 1.0) / 2.0
            if score < 0.6:
                score = 0.0
            video_scores.setdefault(vid, {"title": 0.0, "tags": 0.0, "summary": 0.0})
            video_scores[vid]["tags"] = max(video_scores[vid]["tags"], score)

        for emb, meta in zip(results_st["embeddings"][0], results_st["metadatas"][0]):
            if meta["field"] != "summary":
                continue
            vid = meta["video_id"]
            score = (cosine_similarity(q_st_emb, emb) + 1.0) / 2.0
            if score < 0.6:
                score = 0  # 過濾掉低於 0.6 的相似度
            video_scores.setdefault(vid, {"title": 0, "tags": 0, "summary": 0})
            video_scores[vid]["summary"] = score
        
        for vid, scores in video_scores.items():

            total = (
                0.4 * scores["title"] +
                0.3 * scores["tags"] +
                0.3 * scores["summary"]
            )* weight
            print(f"主題{i}: vid={vid}, title={scores["title"]}, tags={scores["tags"]}, summary={scores["summary"]} => total={total}")
            # 如果 vid 已經存在，就保留分數比較高的
            if vid in final_scores:
                final_scores[vid] = max(final_scores[vid], total)
            else:
                final_scores[vid] = total

    # 轉換成 list 並排序
    sorted_scores = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    top_videos = sorted_scores[:k]
    print("分數",sorted_scores[0][1] )
    # ✅ 檢查最高分是否小於 2.5
    if not sorted_scores or sorted_scores[0][1] < 2.5:
        print("查詢相關度最高分小於 2.5，需要自動下載影片")
        need_download = True
        
    # 取得影片資訊與推薦片段時間（改用 with 方式）
    # 如果分數 小於標準，則需要在背景下載影片
    results = []
    with psycopg2.connect("postgresql://postgres:Functrol@localhost:5432/postgres") as conn:
        with conn.cursor() as cursor:
            for vid , total_score in top_videos:
                # 抓 title, summary, embed_url, tag_names
                cursor.execute("""
                    SELECT 
                        v.title, 
                        v.summary, 
                        v.embed_url,
                        ARRAY(
                            SELECT t.name
                            FROM jsonb_array_elements_text(v.tag_ids) AS tag_id_text
                            JOIN tags t ON t.id = tag_id_text::int
                        ) AS tag_names
                    FROM videos v
                    WHERE v.id = %s
                """, (vid,))
                row = cursor.fetchone()
                title, summary, embed_url, tag_names = row if row else ("", "", "", [])

                # 找最佳起始時間
                start = None
                for query_text in expanded_queries:
                    start = get_best_chunk_start(vid, query_text)
                    if start:
                        break

                # 將 "HH:MM:SS" 轉為秒數
                seconds = 0
                if start:
                    h, m, s = map(float, start.split(":"))
                    seconds = int(h * 3600 + m * 60 + s)

                if embed_url:
                    embed_url = embed_url.split("?")[0] + f"?start={seconds}"

                results.append((total_score, vid, title, summary, embed_url, tag_names))

    return expanded_queries, results, need_download #新增 need_download 回傳

def search_videos_with_vectorDB_for_map(query: str, k=5):
    expanded_queries = [query]
    weights = [5, 4, 3, 2, 2, 2]
    final_scores = {}
    need_download = False
    for i, query_text in enumerate(expanded_queries):
        video_scores = {}
        weight = weights[i]

        q_tt_emb = model_tt.encode([query_text])[0]
        q_st_emb = model_st.encode([query_text])[0]

        # Title 專屬前 10
        results_title = collection_tt.query(
            query_embeddings=[q_tt_emb],
            n_results=10,
            where={"field": "title"},
            include=["metadatas", "embeddings"]
        )

        # 處理 summary
        results_st = collection_st.query(
            query_embeddings=[q_st_emb],
            n_results=10,
            include=["metadatas", "embeddings"]
        )

        # 建立候選影片清單（僅 title 命中的影片）
        candidate_vids = set()
        for emb, meta in zip(results_title["embeddings"][0], results_title["metadatas"][0]):
            vid = meta["video_id"]
            score = (cosine_similarity(q_tt_emb, emb) + 1.0)/2.0
            if score < 0.6:
                score = 0  # 過濾掉低於 0.6 的相似度
            video_scores.setdefault(vid, {"title": 0, "tags": 0, "summary": 0})
            video_scores[vid]["title"] = score
            candidate_vids.add(vid)

        # 只用一個 where 條件：video_id in 候選
        tags_docs = collection_tt.get(
            where={"video_id": {"$in": list(candidate_vids)}},  # 只放「一個」operator
            include=["metadatas", "embeddings"]
        )

        # 對每支影片取「所有 tag 向量」的最大相似度當作該影片 tags 分數
        for emb, meta in zip(tags_docs.get("embeddings", []), tags_docs.get("metadatas", [])):
            if meta.get("field") != "tags":
                continue  # 只算 tags
            vid = meta.get("video_id")
            if not vid:
                continue
            score = (cosine_similarity(q_tt_emb, emb) + 1.0) / 2.0
            if score < 0.6:
                score = 0.0
            video_scores.setdefault(vid, {"title": 0.0, "tags": 0.0, "summary": 0.0})
            video_scores[vid]["tags"] = max(video_scores[vid]["tags"], score)

        for emb, meta in zip(results_st["embeddings"][0], results_st["metadatas"][0]):
            if meta["field"] != "summary":
                continue
            vid = meta["video_id"]
            score = (cosine_similarity(q_st_emb, emb) + 1.0) / 2.0
            if score < 0.6:
                score = 0  # 過濾掉低於 0.6 的相似度
            video_scores.setdefault(vid, {"title": 0, "tags": 0, "summary": 0})
            video_scores[vid]["summary"] = score
        
        for vid, scores in video_scores.items():

            total = (
                0.4 * scores["title"] +
                0.3 * scores["tags"] +
                0.3 * scores["summary"]
            )* weight
            print(f"主題{i}: vid={vid}, title={scores["title"]}, tags={scores["tags"]}, summary={scores["summary"]} => total={total}")
            # 如果 vid 已經存在，就保留分數比較高的
            if vid in final_scores:
                final_scores[vid] = max(final_scores[vid], total)
            else:
                final_scores[vid] = total

    # 轉換成 list 並排序
    sorted_scores = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    top_videos = sorted_scores[:k]
    print("分數",sorted_scores[0][1] )
    if sorted_scores[0][1] < 2.5: #先給1，因為被YT鎖住了
        need_download = True
        
    # 取得影片資訊與推薦片段時間（改用 with 方式）
    # 如果分數 小於標準，則需要在背景下載影片
    results = []
    with psycopg2.connect("postgresql://postgres:Functrol@localhost:5432/postgres") as conn:
        with conn.cursor() as cursor:
            for vid , total_score in top_videos:
                # 抓 title, summary, embed_url, tag_names
                cursor.execute("""
                    SELECT 
                        v.title, 
                        v.summary, 
                        v.embed_url,
                        ARRAY(
                            SELECT t.name
                            FROM jsonb_array_elements_text(v.tag_ids) AS tag_id_text
                            JOIN tags t ON t.id = tag_id_text::int
                        ) AS tag_names
                    FROM videos v
                    WHERE v.id = %s
                """, (vid,))
                row = cursor.fetchone()
                title, summary, embed_url, tag_names = row if row else ("", "", "", [])

                results.append((total_score, vid, title, summary, embed_url, tag_names))

    return expanded_queries, results,need_download


# 用來找目前最新的id(為了自動下載，請查看services的auto_downlaod)
def get_latest_id(): 
    latest_id = 0
    for col in collections:
        name = col.name
        print(f"🔸 Collection 名稱：{name}")

        collection = client.get_collection(name=name)
        try:
            if name == "tag_vocab":
                break
            data = collection.get(include=["metadatas", "documents"])
            latest_id = data["ids"][-1] if data["ids"] else "無資料"
            # 保留數字
            latest_id = re.sub(r'\D', '', latest_id)
            print(f"   最新 ID：{latest_id}")      

        except Exception as e:
            print(f"   ⚠️ 無法讀取該 collection：{e}")
    return latest_id
# 儲存自動摘要(為了自動下載)
def parse_time(ts):
    h, m, s = ts.split(":")
    return timedelta(hours=int(h), minutes=int(m), seconds=float(s))

def find_start_index(transcript_json, target_time):
    """從字幕中找到第一個開始時間 >= target_time 的 index"""
    for i, seg in enumerate(transcript_json):
        if parse_time(seg["start"]) >= parse_time(target_time):
            return i
    return len(transcript_json)
def slice_transcript_block(transcript_json, start_index, max_tokens=500):
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

請回傳**純 JSON 陣列**，不要加上```json或任何Markdown符號。
格式如下：
[
  {{"start": "00:00:00", "end": "00:00:05", "summary": "段落摘要"}}
]
"""
    model_ai = genai.GenerativeModel('gemini-2.5-flash-lite')
    response = model_ai.generate_content(prompt)
    raw_text = response.text if hasattr(response, "text") else response.parts[0].text
    print("📥 Gemini 回傳原文：\n", raw_text[:500])

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
    existing_ids_chunk = set(collection_chunk.get().get("ids", []))

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

# 自動下載摘要處理(services的auto_download有用到，目前問題:collection_vocab顯示沒東西)
def store_emb(latest_id, conn):
    cursor = conn.cursor()

    # === 抓影片基本資料 + 字幕 + tags 陣列 ===
    cursor.execute("""
        SELECT
            v.id AS video_id,
            v.url,
            v.title,
            v.summary,
            v.transcription,
            v.transcription_with_time,
            (
                SELECT ARRAY_AGG(t.name ORDER BY t.name)
                FROM jsonb_array_elements_text(v.tag_ids) AS tag_id_text
                JOIN tags t ON t.id = tag_id_text::int
            ) AS tags_arr
        FROM video_categories vc
        JOIN videos v ON v.id = vc.video_id
        JOIN categories c ON vc.category_id = c.id
        WHERE v.transcription_with_time IS NOT NULL AND v.id > %s
        GROUP BY v.id, v.url, v.title, v.summary, v.transcription, v.transcription_with_time, v.tag_ids
        ORDER BY v.id
    """, (latest_id,))
    rows = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]

    # === 已存在的 ID（避免重複） ===
    existing_ids_tt = set(collection_tt.get().get("ids", []))
    existing_ids_st = set(collection_st.get().get("ids", []))
    existing_ids_chunk = set(collection_chunks.get().get("ids", []))

    # === 欄位對應模型與 collection（不再把合併 tags 當一顆存入） ===
    field_mapping = {
        "title": (collection_tt, model_tt),
        "summary": (collection_st, model_st),
        "transcription": (collection_st, model_st),
    }

    for row in rows:
        row_dict = dict(zip(columns, row))
        video_id = str(row_dict["video_id"])
        url = row_dict["url"]
        title = row_dict["title"] or ""
        t_with_time = row_dict["transcription_with_time"]
        tags_arr = [t.strip() for t in (row_dict.get("tags_arr") or []) if t and t.strip()]

        # === 儲存 title/summary/transcription 向量 ===
        for field, (collection, model) in field_mapping.items():
            content = (row_dict.get(field) or "").strip()
            if not content:
                continue

            uid = f"{video_id}_{field}"
            if (collection is collection_tt and uid in existing_ids_tt) or \
               (collection is collection_st and uid in existing_ids_st):
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
                    "kind": "single" if field == "title" else "doc"
                }]
            )

        # === 每顆 tag 各存一筆 ===
        for idx, tag_text in enumerate(tags_arr):
            uid = f"{video_id}_tag_{idx}"
            if uid in existing_ids_tt:
                # 允許你日後更新策略時避開重複
                continue
            vec = model_tt.encode([tag_text])[0].tolist()
            collection_tt.add(
                documents=[tag_text],
                embeddings=[vec],
                ids=[uid],
                metadatas=[{
                    "video_id": video_id,
                    "field": "tags",
                    "kind": "single",     # 之後查詢可 where={"field":"tags","kind":"single"}
                    "tag": tag_text,
                    "url": url,
                    "title": title
                }]
            )

        # === 處理 Gemini 字幕分段摘要 ===
        try:
            process_subtitles_with_gemini(
                video_id=video_id,
                transcript_json=t_with_time,
                url=url,
                title=title,
                model_st=model_st,
                collection_chunk=collection_chunks
            )
        except Exception as e:
            print(f"❌ 處理 Gemini 分段失敗：video_id={video_id}，錯誤：{e}")

    print("✅ 已成功將所有影片欄位與『每顆 tag』的向量儲存完成！")
    cursor.close()   
