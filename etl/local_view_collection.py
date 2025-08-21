from chromadb import PersistentClient
from chromadb.config import Settings
import json

# 初始化本地 ChromaDB client
client = PersistentClient(path="local_chroma_db", settings=Settings())

# 取得所有 collections
collections = client.list_collections()

print(f"\n📚 總共有 {len(collections)} 個 collections：\n")

for col in collections:
    name = col.name
    print(f"🔸 Collection 名稱：{name}")

    collection = client.get_collection(name=name)
    try:
        data = collection.get(include=["metadatas", "documents"])
        count = len(data["ids"])
        print(f"   - 總筆數：約 {count} 筆")

        for i, _id in enumerate(data["ids"]):
            print(f"     👉 ID: {_id}")
            print(f"        Metadata: {json.dumps(data['metadatas'][i], ensure_ascii=False)}")
            print(f"        Document: {data['documents'][i][:80]}...")  # 只顯示前 80 字

    except Exception as e:
        print(f"   ⚠️ 無法讀取該 collection：{e}")

    print("-" * 50)
