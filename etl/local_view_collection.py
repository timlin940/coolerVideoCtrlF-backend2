from chromadb import PersistentClient
from chromadb.config import Settings
import re
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
        latest_id = data["ids"][-1] if data["ids"] else "無資料"
        # 保留數字
        latest_id = re.sub(r'\D', '', latest_id)
        print(f"   最新 ID：{latest_id}")

    except Exception as e:
        print(f"   ⚠️ 無法讀取該 collection：{e}")
print(latest_id)