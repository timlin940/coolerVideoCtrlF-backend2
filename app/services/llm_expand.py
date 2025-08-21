import os
import json
import torch
import psycopg2
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
import re
from bertopic import BERTopic
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

# 取得 API Key
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("❌ 找不到 GEMINI_API_KEY，請檢查 .env 設定")

# 設定 Gemini API
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-1.5-flash')

# 真正的topic expand，能生成更多input
def generate_related_queries(input_text):
    """
    使用 LLM (Gemini) 做語意相關詞擴展
    """
    # 1️⃣ 保底：原始 query
    related_queries = [input_text]

    # 2️⃣ LLM prompt
    prompt = (
        f"請列出5個與「{input_text}」密切相關的學術主題詞或關鍵詞(並依照相關程度排序)。"
        "請直接用英文詞語，輸出格式為一個Python list，不用包含``` python 或任何額外標記，只需輸出 Python list 即可，例如："
        "['artificial intelligence', 'deep learning', 'data mining', 'neural networks', 'computer science']"
    )

    # 3️⃣ 調用 LLM
    try:
        response = model.generate_content(
            [
                {"role": "user", "parts": [prompt]}
            ],
            generation_config={
                "temperature": 0.2
            }
        )
        output_text = response.text
        print(f"Gemini LLM輸出: {output_text}")  # 打印原始輸出
        # 移除程式碼塊的標記 (如果存在)
        if output_text.startswith("```python") and output_text.endswith("```"):
            output_text = output_text[len("```python"): -len("```")].strip()
        # 移除可能存在的換行符
        output_text = output_text.replace('\n', '')    
        # 嘗試將回傳內容直接 eval 成 list
        expanded_words = eval(output_text)
        print(f"擴展後的詞語: {expanded_words}")  # 打印擴展後的詞語
        if isinstance(expanded_words, list):
            related_queries.extend(expanded_words)
    except Exception as e:
        print(f"Gemini LLM擴展失敗：{e}")
        print(f"錯誤詳情：{e}")  # 打印更詳細的錯誤信息

    # 4️⃣ 去重
    related_queries = list(dict.fromkeys(related_queries))
    return related_queries