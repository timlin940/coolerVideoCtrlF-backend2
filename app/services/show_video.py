from app.services.vectordb_search_for_main import get_best_chunk_start,cosine_similarity  # 確保你引入 LLM 擴展函數
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

    return {
        "video_id": video_id,
        "title": title,
        "summary": summary,
        "url": embed_url,
        "tags": tag_names,
        "categories": category_names
    }

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

            # Step 2: 撈 categories
            cursor.execute("""
                SELECT c.topic
                FROM video_categories vc
                JOIN categories c ON vc.category_id = c.id
                WHERE vc.video_id = %s
            """, (video_id,))
            category_rows = cursor.fetchall()
            category_names = [r[0] for r in category_rows] if category_rows else []

    # Step 3: 查詢最符合 query 的 chunk（在 collection_chunks）
    query_emb = model_st.encode([query])[0]
    res = collection_chunks.get(
        where={"video_id": video_id}, 
        include=["embeddings", "metadatas"]
    )

    if not res["ids"]:
        return {
            "title": title,
            "summary": summary,
            "embed_url": embed_url,
            "tags": tag_names,
            "categories": category_names,
            "matched_segments": []
        }

    # Step 4: 用 cosine similarity 找最相關的前 5 個字幕片段
    scored_chunks = []
    for emb, meta in zip(res["embeddings"], res["metadatas"]):
        score = cosine_similarity(query_emb, emb)
        scored_chunks.append((score, meta))

    top_chunks = sorted(scored_chunks, key=lambda x: x[0], reverse=True)[:5]

    matched_segments = [
        {
            "start": chunk["start"],
            "summary": chunk.get("summary", "")
        }
        for score, chunk in top_chunks
    ]

    return {
        "query": query,
        "video_id": video_id,
        "title": title,
        "summary": summary,
        "url": embed_url,
        "tags": tag_names,
        "categories": category_names,
        "matched_segments": matched_segments
    }
