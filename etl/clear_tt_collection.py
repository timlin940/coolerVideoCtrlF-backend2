# etl/clear_title_tag_collection.py
import os
from chromadb import PersistentClient
from chromadb.config import Settings

# ============== 可調整參數 ==============
CHROMA_PATH = os.getenv("CHROMA_PATH", "local_chroma_db")
COLLECTION_NAME = os.getenv("TITLE_TAG_COLLECTION", "title_tag_emb")
# ======================================

def main():
    client = PersistentClient(path=CHROMA_PATH, settings=Settings(allow_reset=True))
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception as e:
        print(f"❌ 找不到 collection: {COLLECTION_NAME}，錯誤：{e}")
        return

    # 取得所有 IDs
    ids = collection.get().get("ids", [])
    if not ids:
        print(f"⚠️ Collection '{COLLECTION_NAME}' 已經是空的")
        return

    # 刪除所有
    collection.delete(ids=ids)
    print(f"✅ 已清空 Collection '{COLLECTION_NAME}'，共刪除了 {len(ids)} 筆資料")

if __name__ == "__main__":
    main()
