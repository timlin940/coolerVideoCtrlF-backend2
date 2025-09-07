# build_title_tag_collection.py
import os
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
from chromadb.config import Settings

from app.services.db_utils import login_postgresql

# ============== 可調整參數 ==============
CHROMA_PATH = os.getenv("CHROMA_PATH", "local_chroma_db")
COLLECTION_NAME = os.getenv("TITLE_TAG_COLLECTION", "title_tag_emb")
EMB_MODEL_NAME = os.getenv("TITLE_TAG_MODEL", "paraphrase-MiniLM-L6-v2")
# ======================================

def fetch_video_title_and_tags(cursor):
    """
    取回：video_id, url, title, tags_arr
    tags_arr 為依 tag.name 排序後的文字陣列
    """
    cursor.execute("""
        SELECT
            v.id AS video_id,
            v.url,
            v.title,
            (
                SELECT ARRAY_AGG(t.name ORDER BY t.name)
                FROM jsonb_array_elements_text(v.tag_ids) AS tag_id_text
                JOIN tags t ON t.id = tag_id_text::int
            ) AS tags_arr
        FROM video_categories vc
        JOIN videos v ON v.id = vc.video_id
        JOIN categories c ON vc.category_id = c.id
        WHERE v.id <= 1151
        GROUP BY v.id, v.url, v.title, v.tag_ids
        ORDER BY v.id
    """)
    rows = cursor.fetchall()
    cols = [d[0] for d in cursor.description]
    for r in rows:
        d = dict(zip(cols, r))
        yield {
            "video_id": str(d["video_id"]),
            "url": d["url"],
            "title": d["title"] or "",
            "tags_arr": [t.strip() for t in (d.get("tags_arr") or []) if t and t.strip()],
        }

def main():
    # === 初始化 DB ===
    conn = login_postgresql()
    cursor = conn.cursor()

    # === 初始化 Chroma 與 Collection ===
    client = PersistentClient(path=CHROMA_PATH, settings=Settings(allow_reset=True))
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    # === 初始化 Embedding 模型（title 與 tag 用同一顆）===
    model = SentenceTransformer(EMB_MODEL_NAME)

    # === 先抓既有 IDs，避免重複寫入 ===
    existed = set(collection.get().get("ids", []))

    # === 逐筆處理 ===
    inserted_cnt = 0
    skipped_cnt = 0

    for item in fetch_video_title_and_tags(cursor):
        vid = item["video_id"]
        url = item["url"]
        title = (item["title"] or "").strip()
        tags_arr = item["tags_arr"]

        # 1) 寫入 title
        if title:
            uid_title = f"{vid}_title"
            if uid_title in existed:
                skipped_cnt += 1
            else:
                vec = model.encode([title])[0].tolist()
                collection.add(
                    documents=[title],
                    embeddings=[vec],
                    ids=[uid_title],
                    metadatas=[{
                        "video_id": vid,
                        "field": "title",
                        "kind": "single",
                        "url": url,
                        "title": title
                    }]
                )
                inserted_cnt += 1

        # 2) 寫入每一顆 tag
        for idx, tag_text in enumerate(tags_arr):
            uid_tag = f"{vid}_tag_{idx}"
            if uid_tag in existed:
                skipped_cnt += 1
                continue
            vec = model.encode([tag_text])[0].tolist()
            collection.add(
                documents=[tag_text],
                embeddings=[vec],
                ids=[uid_tag],
                metadatas=[{
                    "video_id": vid,
                    "field": "tags",
                    "kind": "single",
                    "tag": tag_text,
                    "url": url,
                    "title": title
                }]
            )
            inserted_cnt += 1

    cursor.close()
    conn.close()
    print(f"✅ 建立完成：collection='{COLLECTION_NAME}'")
    print(f"   新增 {inserted_cnt} 筆，跳過(已存在) {skipped_cnt} 筆。")

if __name__ == "__main__":
    main()
