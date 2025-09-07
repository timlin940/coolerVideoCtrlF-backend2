import psycopg2
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import torch
from app.chroma_client import ChromaDBClient
load_dotenv()   

client = ChromaDBClient.get_instance().get_client()
model_st = SentenceTransformer("BAAI/bge-m3", device='cuda' if torch.cuda.is_available() else 'cpu')
collection_chunks = client.get_or_create_collection(name="transcription_chunks_emb")


def get_video_no_query(video_id: str):
    db_url = os.getenv("DATABASE_URL")
    with psycopg2.connect(db_url) as conn:
        with conn.cursor() as cursor:
            # 撈影片資料 + tags
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
            """, (video_id,))
            row = cursor.fetchone()
            if not row:
                return {}

            title, summary, embed_url, tag_names = row

            # 撈影片對應的 category 名稱們
            cursor.execute("""
                SELECT c.topic
                FROM video_categories vc
                JOIN categories c ON vc.category_id = c.id
                WHERE vc.video_id = %s
            """, (video_id,))
            category_rows = cursor.fetchall()
            category_names = [r[0] for r in category_rows] if category_rows else []
    print("測試3")
    return {
        "video_id": video_id,
        "title": title,
        "summary": summary,
        "url": embed_url,
        "tags": tag_names,
        "categories": category_names
    }

import google.generativeai as genai
import os
import psycopg2
import json

api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-2.5-flash-lite")

def rerank_segments_with_gemini(query: str, segments_data: list) -> list:
    """
    使用 Gemini 對搜尋結果進行重新排序
    
    Args:
        query: 使用者查詢
        segments_data: 包含 summary 和 metadata 的片段列表
    
    Returns:
        重新排序後的片段列表
    """
    if not segments_data:
        return []
    
    # 準備給 Gemini 的 prompt
    segments_text = ""
    for i, segment in enumerate(segments_data):
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
        ranking = result.get("ranking", list(range(1, len(segments_data) + 1)))
        explanation = result.get("explanation", "")
        
        # 根據排序重新組織片段
        reranked_segments = []
        for rank_idx in ranking:
            if 1 <= rank_idx <= len(segments_data):
                segment = segments_data[rank_idx - 1].copy()
                reranked_segments.append(segment)
        
        # 如果有遺漏的片段，補充到後面
        used_indices = set(ranking)
        for i in range(1, len(segments_data) + 1):
            if i not in used_indices:
                reranked_segments.append(segments_data[i - 1])
        
        return reranked_segments
        
    except Exception as e:
        print(f"Gemini 重排序失敗: {e}")
        # 如果 Gemini 失敗，返回原始順序
        return segments_data

def get_video_with_query(video_id: str, query: str):
    db_url = os.getenv("DATABASE_URL")
    with psycopg2.connect(db_url) as conn:
        with conn.cursor() as cursor:
            # Step 1: 撈影片基本資料 + tags
            cursor.execute("""
                SELECT 
                    v.title, 
                    v.summary, 
                    v.embed_url,
                    v.highlights,
                    ARRAY(
                        SELECT t.name
                        FROM jsonb_array_elements_text(v.tag_ids) AS tag_id_text
                        JOIN tags t ON t.id = tag_id_text::int
                    ) AS tag_names
                FROM videos v
                WHERE v.id = %s
            """, (video_id,))
            row = cursor.fetchone()
            if not row:
                return {}

            title, summary, embed_url, highlights, tag_names = row

            # Step 2: 撈 categories
            cursor.execute("""
                SELECT c.topic
                FROM video_categories vc
                JOIN categories c ON vc.category_id = c.id
                WHERE vc.video_id = %s
            """, (video_id,))
            category_rows = cursor.fetchall()
            category_names = [r[0] for r in category_rows] if category_rows else []

    # Step 2.5: 決定 summary 要用哪個欄位
    use_highlights = False
    if query.endswith("|map"):
        use_highlights = True
        query = query.replace("|map", "").strip()

    # Step 3: 使用 query 方法查詢最符合的前 10 個 chunks
    query_emb = model_st.encode([query])[0]
    res = collection_chunks.query(
        query_embeddings=[query_emb],
        n_results=20,
        where={"video_id": video_id},
        include=["metadatas", "distances"]  # 包含距離分數
    )

    if not res["ids"] or not res["ids"][0]:
        return {
            "query": query,
            "video_id": video_id,
            "title": title,
            "summary": highlights if use_highlights else summary,
            "url": embed_url,
            "tags": tag_names,
            "categories": category_names,
            "matched_segments": []
        }

    # Step 4: 整理 query 結果準備給 Gemini 重排序
    # query() 方法返回的是巢狀列表結構 [[...]]
    metadatas = res["metadatas"][0]  # 取第一個查詢的結果
    distances = res["distances"][0] if "distances" in res else []
    
    # 準備給 Gemini 重排序的資料
    segments_for_rerank = []
    for i, meta in enumerate(metadatas):
        segment_data = {
            "start": meta.get("start", 0),
            "summary": meta.get("summary", ""),
        }
        # 如果有距離分數，也保留作為參考
        if distances and i < len(distances):
            segment_data["distance_score"] = distances[i]
        
        segments_for_rerank.append(segment_data)
    
    # 如果沒有找到任何片段，返回空結果
    if not segments_for_rerank:
        return {
            "query": query,
            "video_id": video_id,
            "title": title,
            "summary": highlights if use_highlights else summary,
            "url": embed_url,
            "tags": tag_names,
            "categories": category_names,
            "matched_segments": []
        }

    # Step 5: 使用 Gemini 進行重排序
    reranked_segments = rerank_segments_with_gemini(query, segments_for_rerank)
    print("測試1")
    print("reranked_segments : ",reranked_segments)
    
    # 只取前 5 個最相關的片段
    top_segments = reranked_segments[:5]
    
    matched_segments = [
        {
            "start": segment["start"],
            "summary": segment.get("summary", "")
        }
        for segment in top_segments
    ]
    print("測試2")
    print(
        "query:", query,
        "video_id:", video_id,
        "title:", title,
        "summary:", highlights if use_highlights else summary,
        "url:", embed_url,
        "tags:", tag_names,
        "categories:", category_names,
        "matched_segments:", matched_segments
    )
    return {
        "query": query,
        "video_id": video_id,
        "title": title,
        "summary": highlights if use_highlights else summary,
        "url": embed_url,
        "tags": tag_names,
        "categories": category_names,
        "matched_segments": matched_segments
    }