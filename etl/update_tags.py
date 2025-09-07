# -*- coding: utf-8 -*-
import os
import hashlib
import psycopg2
from chromadb import PersistentClient
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# ====== 可調整參數 ======
DB_DSN = os.getenv("DB_DSN", "postgresql://postgres:Functrol@localhost:5432/postgres")
CHROMA_PATH = os.getenv("CHROMA_PATH", "local_chroma_db")
COLLECTION_TT = "title_tag_emb"
MODEL_TT_NAME = "paraphrase-MiniLM-L6-v2"
BATCH_SIZE = 256
DRY_RUN = False   # True 先看要刪/要新增哪些，不會動資料
# =======================

# === 初始化 ===
client = PersistentClient(path=CHROMA_PATH, settings=Settings(allow_reset=True))
collection_tt = client.get_or_create_collection(name=COLLECTION_TT)
model_tt = SentenceTransformer(MODEL_TT_NAME)

def stable_tag_uid(video_id: str, tag_text: str) -> str:
    """用 tag 文字做穩定 ID（避免用索引位置）。"""
    h = hashlib.md5(tag_text.strip().lower().encode("utf-8")).hexdigest()[:12]
    return f"{video_id}_tag_{h}"

def dedup_preserve(seq):
    seen = set()
    out = []
    for x in seq:
        k = x.strip().lower()
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(x)
    return out

def migrate_tags_to_single_embeddings(conn, dry_run: bool = False, batch_size: int = 256):
    """
    1) 刪除舊的「合併 tags」向量（field='tags' 且 metadata 沒有 'tag' 欄位），
       以及『單顆 tag 但使用舊 ID 規則（非雜湊）』的向量。
    2) 以『每顆 tag 一筆』重建（metadata 不含 kind；只寫入 video_id/field/tag/url/title）。
    """
    cur = conn.cursor()
    cur.execute("""
        SELECT
            v.id   AS video_id,
            v.url,
            v.title,
            (
                SELECT ARRAY_AGG(t.name ORDER BY t.name)
                FROM jsonb_array_elements_text(v.tag_ids) AS tag_id_text
                JOIN tags t ON t.id = tag_id_text::int
            ) AS tags_arr
        FROM videos v
        WHERE v.tag_ids IS NOT NULL
        ORDER BY v.id
    """)
    rows = cur.fetchall()

    ids_to_add, embs_to_add, docs_to_add, metas_to_add = [], [], [], []

    def flush_add_batch():
        nonlocal ids_to_add, embs_to_add, docs_to_add, metas_to_add
        if not ids_to_add:
            return
        if dry_run:
            print(f"[DRY-RUN] Would ADD {len(ids_to_add)} embeddings")
            ids_to_add, embs_to_add, docs_to_add, metas_to_add = [], [], [], []
            return
        collection_tt.add(
            ids=ids_to_add,
            embeddings=embs_to_add,
            documents=docs_to_add,
            metadatas=metas_to_add
        )
        ids_to_add, embs_to_add, docs_to_add, metas_to_add = [], [], [], []

    total_deleted = 0
    total_added = 0

    for video_id, url, title, tags_arr in rows:
        video_id = str(video_id)
        url = url or ""
        title = title or ""
        tags_arr = dedup_preserve([t for t in (tags_arr or []) if t])

        if not tags_arr:
            continue

        # 抓這支影片所有 field='tags' 的向量
        existing = collection_tt.get(
            where={
                "$and": [
                    {"video_id": {"$eq": video_id}},
                    {"field": {"$eq": "tags"}},
                ]
            }
        )        
        ex_ids = existing.get("ids") or []
        ex_metas = existing.get("metadatas") or []

        # 判定要刪除的：沒有 'tag' 欄位（舊合併）、或 'tag' 存在但 id != 雜湊規則（舊單顆但 id 不是雜湊）
        legacy_ids = []
        valid_md5_ids = set()
        for _id, meta in zip(ex_ids, ex_metas):
            meta = meta or {}
            tag_text = meta.get("tag")
            if not tag_text:
                legacy_ids.append(_id)  # 舊合併（沒有 tag 欄位）
                continue
            expected = stable_tag_uid(video_id, tag_text)
            if _id != expected:
                legacy_ids.append(_id)  # 舊單顆但 id 不是雜湊，列為要刪
            else:
                valid_md5_ids.add(_id)

        if legacy_ids:
            if dry_run:
                print(f"[DRY-RUN] Would DELETE legacy ids for video {video_id}: {legacy_ids}")
            else:
                collection_tt.delete(ids=legacy_ids)
            total_deleted += len(legacy_ids)

        # 準備欲存在的（MD5 規則）目標 id
        target_ids = {stable_tag_uid(video_id, t): t for t in tags_arr}

        # 產生待新增清單（僅那些目前不存在的）
        pending_ids = []
        pending_texts = []
        for uid, tag_text in target_ids.items():
            if uid in valid_md5_ids:
                continue  # 已經是新規則，跳過
            pending_ids.append(uid)
            pending_texts.append(tag_text)

        if not pending_texts:
            continue

        # 批次 encode（每支影片先做一批；也可改成跨影片累加到 batch_size 再 encode）
        vecs = model_tt.encode(pending_texts)
        for uid, tag_text, vec in zip(pending_ids, pending_texts, vecs):
            ids_to_add.append(uid)
            embs_to_add.append(vec.tolist())
            docs_to_add.append(tag_text)
            metas_to_add.append({
                "video_id": video_id,
                "field": "tags",
                "tag": tag_text,
                "url": url,
                "title": title
            })
            if len(ids_to_add) >= batch_size:
                flush_add_batch()
                if not dry_run:
                    total_added += batch_size

    # 收尾 flush
    last = len(ids_to_add)
    flush_add_batch()
    if not dry_run:
        total_added += last

    cur.close()
    if dry_run:
        print("✅ DRY-RUN 完成：沒有實際刪除/新增，請檢查上面輸出。")
    else:
        print(f"✅ 轉換完成：刪除 {total_deleted} 筆舊 tags 向量，新增 {total_added} 筆單顆 tag 向量。")

def main():
    with psycopg2.connect(DB_DSN) as conn:
        migrate_tags_to_single_embeddings(conn, dry_run=DRY_RUN, batch_size=BATCH_SIZE)

if __name__ == "__main__":
    main()
