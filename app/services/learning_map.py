import os
import re
from dotenv import load_dotenv
import google.generativeai as genai
from app.services.vectordb_search_for_main import search_videos_with_vectorDB,search_videos_with_vectorDB_for_map

load_dotenv()  # 載入 .env

# 取得 API Key
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("❌ 找不到 GEMINI_API_KEY，請檢查 .env 設定")
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-1.5-flash')


# 還沒機會測試 先擺著XD
def generate_learning_map(input_text):
    prompt = f"""
你是一個教學地圖設計專家，請幫我設計一份完整的學習地圖。

我想學的是：「{input_text}」

請幫我分成三個階段（階段1、2、3），每個階段包含 2～3 個主要學習項目，每個項目再列出 3～5 個小進度（學習細項）。最後，請幫我為每個「項目」提供能夠用來搜尋 YouTube 或 Google 的**一個英文關鍵字（keyword）**。

**請注意：教學內容請用繁體中文撰寫，但每個項目的 Keywords 請全部使用英文。**

請使用這種清楚的階層式排版。內容要具體、有條理、容易懂，並以學生學習的邏輯順序安排。

請依下列格式產出：

階段 1：主題名稱
1. 項目名稱
    - 小進度1
    - 小進度2
    - ...
    - keywords: english keyword
...
"""

    try:
        response = model.generate_content(
            [{"role": "user", "parts": [prompt]}],
            generation_config={"temperature": 0.2}
        )
        full_text = response.text
        print(full_text)
        # 分段：找出所有階段區塊
        phase_blocks = re.split(r"\n(?=階段 \d+：)", full_text)
        phases = {}

        for idx, block in enumerate(phase_blocks, start=1):
            lines = block.strip().splitlines()
            if not lines:
                continue

            phase_title = lines[0].strip()
            items = []

            # 擷取每個項目區塊（以數字加點開頭）
            item_blocks = re.split(r"\n(?=\d+\.\s)", block)
            for item_text in item_blocks[1:]:  # 跳過第一行階段標題
                item_lines = item_text.strip().splitlines()
                if not item_lines:
                    continue

                # 項目標題（例如：1. Python 基礎程式設計）
                title_line = item_lines[0]
                title = re.sub(r"^\d+\.\s*", "", title_line).strip()

                steps = []
                keywords = []

                for line in item_lines[1:]:  # 從第 2 行開始（第 1 行是標題）
                    line_strip = line.strip()

                    # 若是 keywords 行，提取並中止 step 擷取
                    if line_strip.lower().startswith("keywords:") or line_strip.lower().startswith("- keywords:"):
                        # 支援 "keywords:" 或 "- keywords:"
                        kw_line = line_strip.split(":", 1)[-1]
                        keywords = [kw.strip() for kw in kw_line.split(",") if kw.strip()]
                        break

                    # 若是合法步驟行
                    if line_strip.startswith("-"):
                        steps.append(line_strip[1:].strip() if line_strip[1] == " " else line_strip[2:].strip())

                # 搜尋相關影片（若有 keyword）
                video = None
                if keywords:
                    try:
                        expanded, videos = search_videos_with_vectorDB_for_map(query=keywords[0], k=3)
                        video = videos if videos else None
                    except Exception as e:
                        print(f"❌ 搜尋影片失敗：{e}")
                        video = None

                items.append({
                    "title": title,
                    "steps": steps,
                    "keywords": keywords,
                    "video": video
                })

            phases[f"phase_{idx}"] = {
                "title": phase_title,
                "items": items
            }

        return phases

    except Exception as e:
        print(f"❌ Gemini LLM擴展失敗：{e}")
        return None