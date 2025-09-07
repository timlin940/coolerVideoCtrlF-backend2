import os
import json
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
model = genai.GenerativeModel('gemini-2.5-flash-lite')

# 真正的topic expand，能生成更多input
def generate_related_queries(input_text: str):
    """
    使用 LLM (Gemini) 做語意相關詞擴展
    """
   
    prompt1 = (
        f"請將這串輸入{input_text}翻譯成英文，如果已經是英文就保持不變；不要有除了英文以外其他的廢話和符號"
    )
    input_text_en =  response = model.generate_content(
            [{"role": "user", "parts": [prompt1]}],
            generation_config={"temperature": 0.2}
        )
    input_text_en = input_text_en.text
    print(f"翻譯後的輸入: {input_text_en}")

    prompt = (
        f"請列出5個與「{input_text}」密切相關的學術主題詞或關鍵詞(並依照相關程度排序)。"
        "請直接用英文詞語，輸出格式為一個Python list，不用包含``` python 或任何額外標記，只需輸出 Python list 即可，例如："
        "['artificial intelligence', 'deep learning', 'data mining', 'neural networks', 'computer science']"
    )
    related_queries = [input_text_en]
    try:
        response = model.generate_content(
            [{"role": "user", "parts": [prompt]}],
            generation_config={"temperature": 0.2}
        )
        output_text = response.text
        print(f"Gemini LLM輸出: {output_text}")

        if output_text.startswith("```python") and output_text.endswith("```"):
            output_text = output_text[len("```python"):-len("```")].strip()

        output_text = output_text.replace('\n', '')
        expanded_words = eval(output_text)
        print(f"擴展後的詞語: {expanded_words}")

        if isinstance(expanded_words, list):
            related_queries.extend(expanded_words)

    except Exception as e:
        print(f"Gemini LLM擴展失敗：{e}")

    return list(dict.fromkeys(related_queries))