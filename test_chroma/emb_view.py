from chromadb import PersistentClient
from chromadb.config import Settings
import json

# 初始化本地 ChromaDB client
client = PersistentClient(path="test_chroma_db", settings=Settings())

# 指定 video_id 範圍（以字串形式）
video_id_range = set(str(i) for i in range(456, 461))

# 取得所有 collections
collections = client.list_collections()

print(f"\n📦 共找到 {len(collections)} 個 collections：\n")

for col in collections:
    collection_name = col.name
    print(f"🔍 讀取 collection：{collection_name}")
    collection = client.get_collection(name=collection_name)

    try:
        data = collection.get(include=["metadatas", "documents"], limit=99999)
        count = len(data["ids"])
        print(f"   - 總筆數：約 {count} 筆")

        match_count = 0
        for i, _id in enumerate(data["ids"]):
            meta = data["metadatas"][i]
            doc = data["documents"][i]
            vid = str(meta.get("video_id"))

            if vid in video_id_range:
                match_count += 1
                print(f"\n🔹 ID: {_id}")
                print(f"   📌 Metadata: {json.dumps(meta, ensure_ascii=False)}")
                print(f"   📄 Document: {doc[:120]}...")  # 只顯示前 120 字以防爆字

        if match_count == 0:
            print(f"   ⚠️ 沒有符合 video_id {min(video_id_range)}～{max(video_id_range)} 的資料")

    except Exception as e:
        print(f"❌ 無法讀取 collection {collection_name}：{e}")

    print("-" * 60)
