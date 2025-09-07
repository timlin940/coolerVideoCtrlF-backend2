
import os
import psycopg2
import json
import google.generativeai as genai

# ✅ 設定 Gemini
api_key =  os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-2.5-flash-lite")

# ✅ 資料庫連線
DATABASE_URL = (
            os.getenv("DATABASE_URL") 
        )
conn = psycopg2.connect(DATABASE_URL)
cursor = conn.cursor()

# ✅ 取得所有影片中 tag_ids 為空的影片
cursor.execute("SELECT id, summary FROM videos WHERE tag_ids::text = '[]' order by id")
videos = cursor.fetchall()

# ✅ 取得所有現有 tag
cursor.execute("SELECT id, name FROM tags")
existing_tags = cursor.fetchall()
tag_name_to_id = {name: tid for tid, name in existing_tags}

print(f"🔍 共 {len(videos)} 支影片待處理")

# ✅ 處理每部影片
for video_id, summary in videos:
    tag_list = list(tag_name_to_id.keys())

    # 建立 prompt
    prompt = f"""
    你是專業的知識分類專家，請根據下列影片摘要，嚴格遵守規則選出最適合的 3 個英文標籤（用逗號分隔，不得多於 3 個）：

    摘要內容如下：
    「{summary}」

    已知標籤清單如下：
    {', '.join(tag_list)}

    請**務必遵守以下規則**：

    1. 優先從清單中挑選 3 個最貼近摘要主題的標籤。
    2. 若清單中沒有合適的，請**直接創造新的標籤（最多 3 個）**。
    3. **禁止創造與現有標籤語意重複或相近的標籤**（例如已有 Machine Learning，就不能創造 ML 或 AI ML）。
    4. 新標籤必須是**清楚、簡短、意義明確的英文詞彙**，且**限制為兩個單字以內**。
    5. 僅回傳標籤內容，用英文逗號分隔，**不要附加任何文字、引號、格式符號或說明**。

    範例正確回覆格式（僅供參考）：
    AI, Blockchain, Cybersecurity
    """

    try:
        response = model.generate_content(prompt)
        tags_raw = response.text.strip()
        print(f"📌 影片 {video_id} Gemini 回傳 tag：{tags_raw}")
    except Exception as e:
        print(f"❌ 處理影片 {video_id} 發生錯誤：{e}")
        continue

    # 解析 tags
    tag_names = [t.strip().strip(".,，") for t in tags_raw.split(",") if t.strip()]
    tag_ids = []

    for tag in tag_names:
        # 若 tag 已存在
        if tag in tag_name_to_id:
            tag_ids.append(tag_name_to_id[tag])
        else:
            try:
                # 插入新 tag 並取回 id
                cursor.execute("INSERT INTO tags (name) VALUES (%s) RETURNING id", (tag,))
                new_tag_id = cursor.fetchone()[0]
                tag_name_to_id[tag] = new_tag_id  # 更新快取
                tag_ids.append(new_tag_id)
                print(f"✅ 新增 tag：{tag}（ID: {new_tag_id}）")
            except Exception as e:
                print(f"❌ 無法新增 tag {tag}：{e}")

    # ✅ 更新該影片的 tag_ids 欄位
    try:
        tag_ids_json = json.dumps(tag_ids, ensure_ascii=False)
        cursor.execute("UPDATE videos SET tag_ids = %s WHERE id = %s", (tag_ids_json, video_id))
        print(f"🎯 已更新影片 {video_id} 的 tag_ids 為：{tag_ids_json}")
    except Exception as e:
        print(f"❌ 更新影片 {video_id} tag_ids 失敗：{e}")

# ✅ 最後提交更動
conn.commit()
cursor.close()
conn.close()
print("✅ 所有未分配的影片已處理完畢")
