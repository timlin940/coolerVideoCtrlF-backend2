from chromadb import PersistentClient
from chromadb.config import Settings

client = PersistentClient(path="local_chroma_db", settings=Settings())
collections = client.list_collections()

def batch_delete(collection, ids, batch_size=500):
    for i in range(0, len(ids), batch_size):
        batch = ids[i:i+batch_size]
        collection.delete(ids=batch)

for col in collections:
    name = col.name
    collection = client.get_collection(name=name)

    try:
        data = collection.get(include=["metadatas"], limit=99999)
        target_ids = [
            _id for _id, meta in zip(data["ids"], data["metadatas"])
            if meta.get("video_id") and str(meta["video_id"]).isdigit()
        ]

        if target_ids:
            batch_delete(collection, target_ids)
            print(f"✅ 已從 collection {name} 刪除 {len(target_ids)} 筆的資料")
        else:
            print(f"⚠️ Collection {name} 沒有符合 video_id 條件的資料")

    except Exception as e:
        print(f"❌ 清空失敗：{name}，錯誤：{e}")

print("\n🎉 所有符合條件的資料刪除完畢！")