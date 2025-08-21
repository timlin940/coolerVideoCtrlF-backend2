from chromadb import PersistentClient
from chromadb.config import Settings
import json
import torch
from sentence_transformers import SentenceTransformer

# 初始化本地 ChromaDB client
client = PersistentClient(path="test_chroma_db", settings=Settings())

# 指定要查看的 collection 名稱
collection_chunks = client.get_or_create_collection(name="transcription_chunks_emb")
video_id_range = [str(i) for i in range(398, 461)]  # 設定影片查詢範圍

# 載入模型
model_st = SentenceTransformer("BAAI/bge-m3", device='cuda' if torch.cuda.is_available() else 'cpu')

# === 公用 cosine similarity 函數 ===
def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    return dot / (norm_a * norm_b + 1e-9)

# === 查詢字幕片段最相關的起始時間（取前一段） ===
def get_best_chunk_start_and_summary(video_id: str, query: str):
    query_emb = model_st.encode([query])[0]
    res = collection_chunks.get(where={"video_id": video_id}, include=["embeddings", "metadatas", "documents"])
    if not res["ids"]:
        return None, None, None

    best_idx = -1
    best_score = -1

    for i, (emb, meta) in enumerate(zip(res["embeddings"], res["metadatas"])):
        score = cosine_similarity(query_emb, emb)
        if score > best_score:
            best_score = score
            best_idx = i

    if best_idx < 0:
        return None, None, None

    best_meta = res["metadatas"][best_idx]
    best_doc = res["documents"][best_idx]
    start_time = best_meta["start"]
    return start_time, best_doc, best_score

# === 主程式 ===
query = "how to mine BTC ? ?" # 查詢的關鍵字

print(f"\n🔍 根據輸入內容「{query}」搜尋最佳片段...\n")
for vid in video_id_range:
    start_time, summary, score = get_best_chunk_start_and_summary(vid, query)
    if start_time:
        print(f"🎬 Video ID: {vid}")
        print(f"⏱️ 最相關片段起始時間：{start_time}")
        print(f"📝 摘要內容：{summary[:250]}...")
        print(f"📈 相似分數：{score:.4f}")
        print("-" * 60)
    else:
        print(f"⚠️ 無法找到 Video ID {vid} 的相關片段")
