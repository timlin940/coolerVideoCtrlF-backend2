from chromadb import PersistentClient
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import torch
from app.services.llm_expand import generate_related_queries  # 確保你引入 LLM 擴展函數
import psycopg2
import os
from dotenv import load_dotenv
load_dotenv()
from app.chroma_client import ChromaDBClient

# === 初始化 ChromaDB client & collection ===
client = ChromaDBClient.get_instance().get_client()
collection_tt = client.get_or_create_collection(name="title_tag_emb")
collection_st = client.get_or_create_collection(name="summary_transcription_emb")
collection_chunks = client.get_or_create_collection(name="transcription_chunks_emb")

# === 模型 ===
model_tt = SentenceTransformer("paraphrase-MiniLM-L6-v2", device='cuda' if torch.cuda.is_available() else 'cpu')
model_st = SentenceTransformer("BAAI/bge-m3", device='cuda' if torch.cuda.is_available() else 'cpu')

# === 公用 cosine similarity 函數 ===
def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    return dot / (norm_a * norm_b + 1e-9)

# === 查詢字幕片段最相關的起始時間（取前一段） === ###最後return的best_idx有-1才是前一段
def get_best_chunk_start(video_id: str, query: str):
    query_emb = model_st.encode([query])[0]
    res = collection_chunks.get(where={"video_id": video_id}, include=["embeddings", "metadatas"])
    if not res["ids"]:
        return None

    best_idx = -1
    best_score = -1

    for i, (emb, meta) in enumerate(zip(res["embeddings"], res["metadatas"])):
        score = cosine_similarity(query_emb, emb)
        if score > best_score:
            best_score = score
            best_idx = i

    if best_idx < 0:
        return None
    if best_idx == 0:
        return res["metadatas"][0]["start"]
    return res["metadatas"][best_idx]["start"]



# === 主流程 ===
def search_videos_with_vectorDB(query: str, k=5):
    expanded_queries = generate_related_queries(query)
    video_scores = {}
    weights = [0.3, 0.25, 0.15, 0.1, 0.1, 0.1]

    for i, query_text in enumerate(expanded_queries):
        weight = weights[i]

        q_tt_emb = model_tt.encode([query_text])[0]
        q_st_emb = model_st.encode([query_text])[0]

        # 處理 title/tags
        results_tt = collection_tt.query(
            query_embeddings=[q_tt_emb],
            n_results=10,
            include=["metadatas", "embeddings"]
        )

        for emb, meta in zip(results_tt["embeddings"][0], results_tt["metadatas"][0]):
            vid = meta["video_id"]
            field = meta["field"]  # title or tags
            if field not in ["title", "tags"]:
                continue
            score = cosine_similarity(q_tt_emb, emb) + 1
            video_scores.setdefault(vid, {"title": 0, "tags": 0, "summary": 0})
            video_scores[vid][field] += score * weight

        # 處理 summary
        results_st = collection_st.query(
            query_embeddings=[q_st_emb],
            n_results=10,
            include=["metadatas", "embeddings"]
        )

        for emb, meta in zip(results_st["embeddings"][0], results_st["metadatas"][0]):
            if meta["field"] != "summary":
                continue
            vid = meta["video_id"]
            score = cosine_similarity(q_st_emb, emb) + 1
            video_scores.setdefault(vid, {"title": 0, "tags": 0, "summary": 0})
            video_scores[vid]["summary"] += score * weight

    # 加權總分
    final_scores = []
    for vid, scores in video_scores.items():
        total = (
            0.4 * scores["title"] +
            0.3 * scores["tags"] +
            0.3 * scores["summary"]
        )
        final_scores.append((total, vid))

    final_scores.sort(reverse=True)
    top_videos = final_scores[:k]

    # 取得影片資訊與推薦片段時間（改用 with 方式）
    results = []
    with psycopg2.connect("postgresql://postgres:Functrol@localhost:5432/postgres") as conn:
        with conn.cursor() as cursor:
            for total_score, vid in top_videos:
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

    return expanded_queries, results



def search_videos_with_vectorDB_for_map(query: str, k=5):
    expanded_queries = [query]
    video_scores = {}

    for query_text in expanded_queries:
        q_tt_emb = model_tt.encode([query_text])[0]

        # 使用 ChromaDB 的 query 查詢 top k 筆，限制 field = "title"
        results = collection_tt.query(
            query_embeddings=[q_tt_emb],
            n_results=10,  # 每個 query 找前10個 title
            where={"field": "title"},  # 只找 title 向量
            include=["metadatas", "embeddings"]
        )

        # 統計分數（cosine similarity * 1）→ 可視需要乘以權重
        for emb, meta in zip(results["embeddings"][0], results["metadatas"][0]):
            vid = meta["video_id"]
            score = cosine_similarity(q_tt_emb, emb) + 1  # 保證為正
            video_scores.setdefault(vid, 0)
            video_scores[vid] += score  # 若未來有多 query，可加上權重

    # 排序影片分數
    final_scores = sorted([(score, vid) for vid, score in video_scores.items()], reverse=True)
    top_videos = final_scores[:k]

    # 取得影片資訊與片段推薦
    results = []
    with psycopg2.connect("postgresql://postgres:Functrol@localhost:5432/postgres") as conn:
        with conn.cursor() as cursor:
            for total_score, vid in top_videos:
                cursor.execute("SELECT title, summary, embed_url FROM videos WHERE id = %s", (vid,))
                row = cursor.fetchone()
                title, summary, embed_url = row if row else ("", "", "")

                start = None
                for query_text in expanded_queries:
                    start = get_best_chunk_start(vid, query_text)
                    if start:
                        break

                # 時間轉換為秒數
                seconds = 0
                if start:
                    h, m, s = map(float, start.split(":"))
                    seconds = int(h * 3600 + m * 60 + s)

                if embed_url:
                    embed_url = embed_url.split("?")[0] + f"?start={seconds}"

                results.append((total_score, vid, title, summary, embed_url))

    return expanded_queries, results