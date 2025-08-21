import random
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.db import get_db
#from app.services.yt_utils import search_youtube_with_subtitles, download_and_save_to_postgresql
from app.services.vectordb_search_for_main import search_videos_with_vectorDB
from app.services.llm_expand import generate_related_queries
from app.services.learning_map import generate_learning_map
from app.services.db_utils import login_postgresql
from app.services.show_video import get_video_with_query, get_video_no_query
from app.chroma_client import ChromaDBClient
from typing import Any, Dict, List, Optional
from fastapi import Query
from fastapi import HTTPException

from datetime import datetime

#後端新增一個解密 JWT 的函數（用於後續需要身份的 API）
from fastapi import Request
from jose import jwt#要加入requirement.txt
from datetime import timedelta
from pydantic import BaseModel
# from app.dependencies import get_current_user, login_postgresql
import json

router = APIRouter()
import os
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"

'''
@router.get("/videos")
def read_videos(db: Session = Depends(get_db)):
    videos = db.execute("SELECT * FROM videos").fetchall()
    return videos
'''
def get_current_user(request: Request):#用來解碼前端傳進來的token
    token = request.headers.get("authorization")
    print("token:", token)
    print("SECRET_KEY:", SECRET_KEY)
    if not token or not token.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")

    try:
        payload = jwt.decode(token[7:], SECRET_KEY, algorithms=[ALGORITHM])
        return payload["user_id"]
    except Exception:
        raise HTTPException(status_code=401, detail="Token decode failed")

### 目前這get /search是要傳回去推薦影片的，但還沒有寫:(
@router.get("/search")
async def search_videos(query: Optional[str] = Query(None)):
    try:
        if query:
            # 有查詢字 → 搜尋影片
            expanded_queries = generate_related_queries(query)
            _, results = search_videos_with_vectorDB(query, k=5)
        else:
            # 沒查詢字 → 推薦影片（你要實作這個 function）
            expanded_queries = []
            #results = get_recommended_videos() 未來要推薦影片函式
            results = []

        response = {
            "query": query,
            "expanded_queries": expanded_queries,
            "results": [
                {
                    "score": score,
                    "video_id": vid,
                    "title": title,
                    "summary": summary,
                    "url": embed_url,
                    "tags": tag_names if tag_names else []  # 確保 tags 是 list
                } 
                for score, vid, title, summary, embed_url,tag_names in results
            ]
        }

        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"搜尋失敗: {str(e)}")

@router.get("/videos/{video_id}")
async def get_video_details(video_id: int, query: Optional[str] = Query(None)):
    try:
        if query:
            video_data = get_video_with_query(str(video_id), query)
        else:
            video_data = get_video_no_query(str(video_id))

        print(video_data)
        return video_data  # 直接回傳 dict 格式
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"取得影片詳情失敗: {str(e)}")


def save_learning_map_to_db(new_map_id, conn, cur, user_id: int, query: str, learning_map: dict):
    for phase_idx, (phase_key, phase) in enumerate(sorted(learning_map.items()), start=1):
        phase_title = phase.get("title", "")
        items = phase.get("items", [])
        
        # 插入筆記資料
        cur.execute("""
            INSERT INTO map_notes (map_id, user_id, note, updated_at)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (user_id, map_id)
            DO UPDATE SET note = EXCLUDED.note, updated_at = EXCLUDED.updated_at;
        """, (new_map_id, user_id, "", datetime.utcnow()))

        for item in items:
            item_title = item.get("title", "")
            steps_raw = item.get("steps", [])
            keywords_raw = item.get("keywords", [])
            video = item.get("video", [])

            # ✅ 強制轉成乾淨 list of string，防止是錯誤字串或 set
            def clean_list(data):
                if isinstance(data, str):
                    # 若是以 {} 包起來的錯誤格式字串 → 轉 list
                    parts = data.strip("{}").split(",")
                    return [s.strip(' "\'') for s in parts if s.strip()]
                elif isinstance(data, (set, tuple)):
                    return [str(s).strip(' "\'') for s in data]
                elif isinstance(data, list):
                    return [str(s).strip(' "\'') for s in data]
                else:
                    return []

            clean_steps = clean_list(steps_raw)
            clean_keywords = clean_list(keywords_raw)

            # ✅ 序列化成 JSON 字串儲存
            clean_steps_json = json.dumps(clean_steps, ensure_ascii=False)
            clean_keywords_json = json.dumps(clean_keywords, ensure_ascii=False)

            video_list = []
            if isinstance(video, list):
                for v in video:
                    if isinstance(v, (list, tuple)) and len(v) > 4:
                        video_list.append({
                            "video_id": v[1],
                            "title": v[2],
                            "summary": v[3],
                            "url": v[4]
                        })
            
            video_info_json = json.dumps(video_list, ensure_ascii=False)


            # ✅ 插入 PostgreSQL，指定為 jsonb
            cur.execute("""
                INSERT INTO learning_map (
                    query, user_id, map_id, phase_number, phase_title, item_title,
                    step_list, keyword_list, video_info, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s);
            """, (
                query, user_id, new_map_id, phase_idx, phase_title, item_title,
                clean_steps_json, clean_keywords_json,  video_info_json, datetime.utcnow()
            ))
        
    conn.commit()

@router.get("/learning_map")
async def get_learning_map(query: Optional[str] = Query(None),user_id: int = Depends(get_current_user)):
    try:
        if not query:
            # 如果沒有提供 query，就回傳空的學習地圖結構
            return {
                "query": None,
                "learning_map": {}
            }
        # 如果學習地圖名稱已經存在
        conn = login_postgresql()
        cur = conn.cursor()
        # query 轉小寫
        query = query.lower() 
        cur.execute("SELECT map_id FROM learning_map WHERE user_id = %s AND query = %s", (user_id, query))
        existing_map = cur.fetchone()
        if existing_map:
            print("學習地圖已存在，map_id:", existing_map[0])
            return{
                "message": "學習地圖已存在",
                "map_id": existing_map[0]
            }

        learning_map = generate_learning_map(query)

        if not learning_map:
            raise HTTPException(status_code=404, detail="無法生成學習地圖")
        
        # 在儲存一張地圖前，先取得新 map_id
        cur.execute("SELECT COALESCE(MAX(map_id), 0) FROM learning_map WHERE user_id = %s", (user_id,))
        current_max_map_id = cur.fetchone()[0]
        new_map_id = current_max_map_id + 1
        # 儲存到資料庫
        save_learning_map_to_db(new_map_id,conn,cur,user_id=user_id, query=query, learning_map=learning_map)
        cur.close()
        conn.close()
        return {
            "message":"成功儲存並製作學習地圖",
            "query": query,
            "learning_map": learning_map,
            "map_id": new_map_id
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成學習地圖失敗: {str(e)}")

#拿取學習地圖
import ast
@router.get("/show_learning_map")
async def show_learning_map(user_id: int = Depends(get_current_user)):
    import ast

    conn = login_postgresql()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM learning_map WHERE user_id = %s", (user_id,))
    rows = cursor.fetchall()

    result_dict = {}

    for row in rows:
        (
            _id, user_id, map_id, phase_number, phase_title,
            item_title, step_list, keyword_list,
             created_at, query,video_info,
        ) = row
        
        try:
            steps = ast.literal_eval(step_list or "[]")
            if not isinstance(steps, list):
                steps = []
        except Exception:
            steps = []

        try:
            keywords = ast.literal_eval(keyword_list or "[]")
            if not isinstance(keywords, list):
                keywords = []
        except Exception:
            keywords = []

        # 處理 video_info 欄位（JSON 字串 → list of dict）
        # try:
        #     videos = json.loads(video_info) if video_info else []
        #     if not isinstance(videos, list):
        #         videos = []
        # except Exception:
        #     videos = []
        import json

        try:
            if isinstance(video_info, str):
                videos = json.loads(video_info)
            elif isinstance(video_info, list):
                videos = video_info
            else:
                videos = []
        except Exception:
            videos = []


        # 如果 query 不在 dict 裡，就初始化並記錄 map_id
        if query not in result_dict:
            result_dict[query] = {
                "map_id": map_id,
                "phases": {}
            }

        phase_key = f"phase_{phase_number}"

        if phase_key not in result_dict[query]["phases"]:
            result_dict[query]["phases"][phase_key] = {
                "title": phase_title,
                "items": []
            }

        parsed_videos = []
        for v in videos:
            if isinstance(v, dict):
                parsed_videos.append({
                    "url": v.get("url", ""),
                    "title": v.get("title", ""),
                    "summary": v.get("summary", ""),
                    "video_id": v.get("video_id", ""),
                })
            elif isinstance(v, list) and len(v) >= 5:
                parsed_videos.append({
                    "url": v[4],
                    "title": v[2],
                    "summary": v[3],
                    "video_id": v[0],
                })

        result_dict[query]["phases"][phase_key]["items"].append({
            "title": item_title,
            "steps": steps,
            "keywords": keywords,
            "video": parsed_videos
        })
    conn.close()
    # 組成 list 格式回傳
    final_result = [
        {
            "query": query,
            "map_id": data["map_id"],
            "learning_map": data["phases"]
        }
        for query, data in result_dict.items()
    ]

    # print(json.dumps(final_result, indent=2, ensure_ascii=True))
    return final_result

@router.delete("/delete_learning_map")
async def delete_learning_map( map_id: int = Query(...)  ,user_id: int = Depends(get_current_user)):#這邊的map_id是前端傳過來的
    conn = login_postgresql()
    cursor = conn.cursor()
    
    cursor.execute("DELETE FROM learning_map WHERE user_id = %s AND map_id = %s ",(user_id,map_id))#刪除地圖資訊
    cursor.execute("DELETE FROM map_notes WHERE user_id = %s AND map_id = %s ",(user_id,map_id))#刪除對應地圖的筆記
    cursor.execute("DELETE FROM map_exam where user_id = %s AND map_id = %s",(user_id,map_id))#刪除對應的測驗問題
    conn.commit()
    conn.close()
    return{
        "message": f"成功刪除id為{map_id}的學習地圖",
        "map_id": map_id
    }

# 顯示學習地圖筆記
@router.get("/get_notes")
async def get_notes(
    map_id: int = Query(..., description="學習地圖 ID"),
    user_id: int = Depends(get_current_user)
):
    conn = login_postgresql()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT note FROM map_notes 
        WHERE user_id = %s AND map_id = %s
    """, (user_id, map_id))
    
    note_row = cursor.fetchone()
    conn.close()
    note_row =list(note_row)
    # print("note_row:", note_row)
    if note_row:
        return {"note": note_row}
    else:
        return {"note": ""}

# 新增note的API
class NoteRequest(BaseModel):
    map_id: int
    note: str

@router.post("/add_note")
async def add_note(
    data: NoteRequest,
    user_id: int = Depends(get_current_user)
):
    map_id = data.map_id
    note = data.note
    conn = login_postgresql()
    cursor = conn.cursor()
    # 插入或更新筆記
    cursor.execute("""
        UPDATE map_notes
        SET note = %s, updated_at = %s
        WHERE user_id = %s AND map_id = %s           
        """
        , (note, datetime.utcnow(), user_id, map_id))

    conn.commit()
    cursor.close()
    conn.close()

    return {"message": "筆記已成功儲存", "map_id": map_id}

@router.get("/videos")
def get_all_videos():
    conn = login_postgresql()
    cursor = conn.cursor()
    cursor.execute("SELECT id, title, duration_str FROM videos")
    data = cursor.fetchall()
    conn.close()
    return {"videos": data}

@router.get("/topics")
def get_all_topics():
    conn = login_postgresql()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM categories")
    data = cursor.fetchall()
    conn.close()
    return {"topics": data}

@router.get("/video-to-topic")
def get_video_topic_relations():
    conn = login_postgresql()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM video_categories")
    data = cursor.fetchall()
    conn.close()
    return {"relations": data}


@router.get("/video-embeddings")
def get_video_chunk_counts():
    # 1. 拿到 collection
    client = ChromaDBClient.get_instance().get_client()
    col_chunk = client.get_collection("transcription_chunks_emb")

    # 2. 抓出所有 metadatas
    chunks = col_chunk.get(include=["metadatas"])
    
    # 3. 統計每個 video_id 有幾個 chunk
    count_map = {}
    for metadata in chunks.get("metadatas", []):
        video_id = metadata.get("video_id")
        if video_id:
            count_map[video_id] = count_map.get(video_id, 0) + 1

    # 4. 組成回傳格式
    result = [
        {"video_id": vid, "chunk_count": count}
        for vid, count in count_map.items()
    ]

    return {
        "total_videos": len(result),
        "videos": result
    }

class RegisterRequest(BaseModel):
    user_name: str
    email: str
    password: str

@router.post("/user_register")#之後改回post，前端傳入帳密
def user_register(data: RegisterRequest):
    user_name = data.user_name
    email = data.email
    password = data.password
    # 前端傳入名稱、信箱、密碼
    conn = login_postgresql()  # 呼叫函數
    cursor = conn.cursor()
    now = datetime.now()
    # user_name = "TimLin" #先預設 之後改
    # email = 'aa0909095679@gmail.com'
    # password = '000'
    # user_name = "qq" #先預設 之後改
    # email = 'qq@gmail.com'
    # password = 'qq'
    try:
        # 檢查 email 是否已存在
        cursor.execute("SELECT id FROM users WHERE email = %s;", (email,))
        result = cursor.fetchone()
        if result is not None:
            return {"status": "Email already registered"}

        # 寫入新使用者
        cursor.execute("""
            INSERT INTO users (username, email, password_hash, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s);
        """, (user_name, email, password, now, now))

        conn.commit()
        return {"status": "User registered successfully"}

    except Exception as e:
        return {"status": "Error", "message": str(e)}

    finally:
        cursor.close()
        conn.close()

# 使用者登入(已經成功登入timlin)
class LoginRequest(BaseModel):
    user_name: str
    email: str
    password: str

@router.post("/user_login")
def user_login(data: LoginRequest):#data: LoginRequest
    user_name = data.user_name
    email = data.email
    password = data.password
    #前端傳入名稱、信箱、密碼
    conn = login_postgresql()
    cursor = conn.cursor()
    # user_name = 'TimLin'#先預設 之後改
    # email = "aa0909095679@gmail.com"
    # password = '000'
    try:
        # 手動設定try catch條件
        try:
            if user_name == "" or email == "" or password == "":
                raise ValueError("Missing required fields")
        except ValueError as ve:
            print("ValueError:", ve)
            return {"status": "Error", "message": str(ve)}

        # 查詢確認資訊是否符合
        cursor.execute("""
            SELECT id FROM users 
            WHERE email = %s AND password_hash = %s AND username = %s;
        """, (email, password, user_name))
        
        result = cursor.fetchone()
        if result is None:
            return {"status": "Login failed. Check credentials."}
        
        user_id = result[0]
        # 產生 token（有效期 7 天）
        payload = {
            "user_id": user_id,
            "exp": datetime.utcnow() + timedelta(days=7)
        }
        token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
        print("token:", token)
        return {"status": "User login successfully",
                #  "access_token": token#回傳前端token
                 "access_token": f"Bearer {token}"
            }

    except Exception as e:
        return {"status": "Error", "message": str(e)}

    finally:
        cursor.close()
        conn.close()

class ClickVideoData(BaseModel):
    video_id: int
    watched_from_sec: int = 0
    watched_to_sec: int = 0

#記錄點下影片的資訊，需要判斷是誰、哪一部影片、從哪看到哪
@router.post("/click_video")
def click_video(data: ClickVideoData,user_id: int = Depends(get_current_user)):
    conn = login_postgresql()
    cursor = conn.cursor()
    print("user_id:", user_id)
    print("video_id:", data.video_id)
    cursor.execute("""
        INSERT INTO user_video_history (user_id, video_id, watched_from_sec, watched_to_sec, date_time)
        VALUES (%s, %s, %s, %s, NOW())
    """, (user_id,  data.video_id, data.watched_from_sec, data.watched_to_sec))

    conn.commit()
    conn.close()

    return {"message": "Click recorded"}

@router.get("/recommend")
def recommend(user_id: int = Depends(get_current_user)):
    conn = login_postgresql()
    cursor = conn.cursor()

   #1. 根據date_time取得user最近10筆的觀看紀錄
    cursor.execute(""" 
        SELECT h.video_id
        FROM user_video_history h
        WHERE h.user_id = %s
        ORDER BY h.date_time DESC
        LIMIT 5
        """, (user_id,))     
    recent_video_ids = [row[0] for row in cursor.fetchall()]

    #2. 根據最近的影片id，找出他們的tags
    if not recent_video_ids:
        return {"message": "沒有最近的觀看紀錄"}
    cursor.execute("""
        SELECT v.tag_ids
        FROM videos v
        WHERE v.id IN %s    
    """, (tuple(recent_video_ids),))
    raw_tag = cursor.fetchall() #這5部影片的tags
    tag_rows = []
    for row in raw_tag:
        if row[0]:  # 確保 tag_ids 不為 None
            tag_ids = row[0] # 這是list
            tag_ids_list = [tag_id for tag_id in tag_ids]  
            tag_rows.extend(tag_ids_list)  # 將每個 tag_id 加入到 tag_rows 列表中
    tag_rows = list(set(tag_rows))  # 去除重複的 tag_ids
    # 3. 依據 tags 找出相關的影片
    cursor.execute("""
        SELECT v.id, v.title,v.embed_url
        FROM videos v
        WHERE EXISTS (
        SELECT 1
        FROM jsonb_array_elements_text(v.tag_ids) AS tag_id
        WHERE tag_id::int = ANY(%s)
        )
        LIMIT 100  
    """, (tag_rows,)) 

    recommended_videos = cursor.fetchall()  # 這些影片的id、title、summary、embed_url
    # 4.從100部影片找隨機10部
    if not recommended_videos:
        return {"message": "沒有相關的推薦影片"}
    random_videos = random.sample(recommended_videos, min(10, len(recommended_videos)))
    conn.close()
    
    return {
        "message": "推薦影片成功",
        "videos": [
            {
                "video_id": vid[0],
                "title": vid[1],
                "embed_url": vid[2]
            } for vid in random_videos
        ]
    }

import google.generativeai as genai
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
import re
api_key =  os.getenv("GEMINI_API_KEY")

genai.configure(api_key=api_key)

#驗收學習成果，前端按下驗收按鈕，根據特定學習地圖生成問題和輸入框
@router.get("/generate_questions")
def generate_questions(
    map_id: int = Query(..., description="學習地圖 ID"),
    phase_str: str = Query(...,description="地圖中的階段phase_NUMBER"),
    user_id: int = Depends(get_current_user)
):
    
    conn = login_postgresql()
    cursor = conn.cursor()
    try:
        # # 原本的
        # cursor.execute(
        #     "SELECT v.id, v.transcription FROM videos v JOIN learning_map l ON v.title = l.video_title WHERE l.user_id = %s AND l.map_id = %s",
        #     (user_id, map_id)
        # )

        # 這段無法執行成功是因為 "v.title = l.item_title"，item_title是由gemini生成的，不是影片title，應該要先從video_info中取出一張地圖中有那些v_id，才能跟video table做join
        # cursor.execute(
        #     "SELECT v.id, v.transcription FROM videos v JOIN learning_map l ON v.title = l.item_title WHERE l.user_id = %s AND l.map_id = %s",
        #     (user_id, map_id)
        # )
        match phase_str:
            case "phase_1":
                phase_number = 1
            case "phase_2":
                phase_number = 2
            case "phase_3":
                phase_number = 3
        print("map_id ",map_id," user_id ",user_id," phase_number ",phase_number)
        # 先取出一張地圖中對應phase_id的的所有v.id
        cursor.execute("""
                select l.video_info FROM learning_map l
                    where l.user_id = %s AND l.map_id = %s AND l.phase_number = %s
                       """, (user_id,map_id,phase_number,))
        video_info = cursor.fetchall()
        vids = []
        for row in video_info:     # 每筆 row 是 tuple，像這樣：([{...}, {...}],)
            video_list = row[0]    # 把 tuple 裡的 list 取出
            for video in video_list:
                vids.append(int(video["video_id"]))
        print(vids)
        # cursor.execute(
        #     """
        #     SELECT DISTINCT v.id, v.transcription
        #     FROM videos v
        #     JOIN learning_map l
        #     ON v.embed_url = (l.video_info->0->>'url')
        #     WHERE l.user_id = %s AND l.map_id = %s
        #     """,
        #     (user_id, map_id)
        # )

        #接下來用我們找到的vids抓取影片字幕

        cursor.execute("""
            SELECT v.transcription FROM videos v
            WHERE v.id IN %s
        """, (tuple(vids),))
        trans = cursor.fetchall()

        print(" 擷取影片筆數：", len(trans))
        
    except Exception as e:
        conn.close()
        return {"message": "抓字幕失敗", "error": str(e)}
    finally:
        cursor.close()
        conn.close()

    
    model = genai.GenerativeModel("gemini-1.5-flash")
    #原始的 prompt設計，題目太簡單沒深度
    """prompt = (
        "請你根據以下影片字幕內容，設計 **10 題繁體中文選擇題**，來檢驗使用者是否理解影片內容。\n"
        "每一題請包含：\n"
        "- `question`：問題文字\n"
        "- `options`：選項列表（4 個選項，包含 1 個正確 + 4 個錯誤，不要總是將答案放在首位）\n"
        "- `answer`：正確答案在 options 中\n\n"
        "**請回傳純 JSON 陣列，不要加上說明或 Markdown 格式。只回傳以下格式：**\n"
        "[\n"
        "  {\n"
        "    \"question\": \"區塊鍊是如何保證資料不被竄改？\",\n"
        "    \"options\": [\n"
        "      \"透過雜湊函數加密並連結區塊\",\n"
        "      \"使用人工審核方式確保資料正確性\",\n"
        "      \"透過雲端儲存資料防止遺失\",\n"
        "      \"每次交易都經由政府備案\"\n"
        "    ],\n"
        "    \"answer\": \"透過雜湊函數加密並連結區塊\"\n"
        "  },\n"
        "  {...共10題}\n"
        "]\n\n"
        f"以下是影片字幕內容：\n{trans}\n"
    )"""
    # 新的 prompt 設計，讓問題更有深度
    prompt = (
        "你是一位專業的教育工作者，擅長出題讓學生思考，而不是只有簡單的填空題，有時有困難的數學計算以及容易搞混的變化題目，並且每次的考卷都不一樣。"
        "請根據以下影片字幕內容，設計 **10 題繁體中文選擇題**，來檢驗使用者是否理解影片內容，但是出題時不要問單純的問影片講甚麼，而是問知識的重點。\n"
        "題目順序隨機打亂，並且每次出題都不一樣。\n"
        "每一題請包含：\n"
        "- `question`：問題文字\n"
        "- `options`：選項列表（4 個選項，包含 1 個正確 + 3 個錯誤，不要總是將答案放在首位）\n"
        "- `answer`：正確答案在 options 中\n"
        "- `explanation`: 請在每題後面加上解釋，為什麼這是正確答案，並對一些錯誤選項作出修正。\n\n"
        "**請回傳純 JSON 陣列，不要加上說明或 Markdown 格式。\n"
         f"以下是影片字幕內容：\n{trans}\n"
            )
    try:
        questions = model.generate_content(prompt)
        raw_text = questions.text.strip()
        if raw_text.startswith("```json"):
            raw_text = re.sub(r"^```json\s*", "", raw_text)
        if raw_text.endswith("```"):
            raw_text = re.sub(r"\s*```$", "", raw_text)
        try:
            questions = json.loads(raw_text)
            return {
            "map_id": map_id,
            "phase_number":phase_number,
            "questions": questions
            }
        except Exception as e:
            return {
                "error": "❌ Gemini 回傳的格式不是純 JSON 陣列",
                "raw": raw_text[:1000],  # 可選：只顯示前 1000 字方便 debug
                "detail": str(e)
            }
    except Exception as e:
        return {"message": "生成問題失敗", "error": str(e)}

# 新增AI問答的區塊，需要讓 AI 知道影片內容
@router.get("/ask_ai")
def ask_ai(
    map_id: int =Query(..., description="學習地圖 ID"),
    question: str = Query(..., description="使用者提問"),
    phase_str:str = Query(...,description="階段NUMBER"),
    user_id: int = Depends(get_current_user)):
    print("資料輸入成功")

    match phase_str:
            case "phase_1":
                phase_number = 1
            case "phase_2":
                phase_number = 2
            case "phase_3":
                phase_number = 3

    model = genai.GenerativeModel("gemini-1.5-flash")
    conn = login_postgresql()
    cursor = conn.cursor()
    # 先取出一張地圖中對應phase_id的的所有v.id
    cursor.execute("""
            select l.video_info FROM learning_map l
                where l.user_id = %s AND l.map_id = %s AND l.phase_number = %s
                    """, (user_id,map_id,phase_number,))
    video_info = cursor.fetchall()
    vids = []
    for row in video_info:     # 每筆 row 是 tuple，像這樣：([{...}, {...}],)
        video_list = row[0]    # 把 tuple 裡的 list 取出
        for video in video_list:
            vids.append(int(video["video_id"]))


    #接下來用我們找到的vids抓取影片字幕
    cursor.execute("""
        SELECT v.transcription FROM videos v
        WHERE v.id IN %s
    """, (tuple(vids),))
    trans = cursor.fetchall()

    conn.close()
    prompt = f"""
            你是一位知識淵博的 AI 助手，使用者會問你許多關於影片的問題，請根據以下影片字幕{trans}問題給出清楚、簡潔的答案。\n
            請回答以下問題：{question}\n\n請給出清楚、簡潔的答案，用繁體中文回答。
            如果user問的知識不在影片內容中，請回答 **抱歉，我無法回答這個問題，因為影片中沒有相關內容。** \n
            """
    try:
        response = model.generate_content(prompt)
        answer = response.text.strip()
        print("AI 回答：", answer)
        return {
            "user_id": user_id,
            "question": question,
            "answer": answer
        }
    except Exception as e:
        return {"message": "AI 回答失敗", "error": str(e)}
    
# 新增學習筆記的區塊:可以在影片下方新增、保存筆記，會隨著map_id儲存、刪除
# @router.post("/add_note")

class QuestionItem(BaseModel):
    question_number: int
    question: str
    options: dict   # 對應到 jsonb
    answer: str
    user_answer: str
    correct: bool
    explanation: str

class ExamSubmission(BaseModel):
    map_id: int
    phase_number: int
    questions: List[QuestionItem]

@router.post("/exam_score")
def submit_exam(
    submission: ExamSubmission,
    user_id: int = Depends(get_current_user)
):
    conn = login_postgresql()
    cursor = conn.cursor()

    for q in submission.questions:
        cursor.execute("""
            INSERT INTO map_exam (
                user_id, map_id, phase_number,
                question_number, question, options,
                answer, user_answer, correct, explanation
            ) VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s)
        """, (
            user_id,
            submission.map_id,
            submission.phase_number,
            q.question_number,
            q.question,
            json.dumps(q.options, ensure_ascii=False),
            q.answer,
            q.user_answer,
            q.correct,
            q.explanation
        ))

    conn.commit()
    conn.close()
    return {"message": "Exam submission saved successfully."}

# 收藏影片的區塊，顯示、新增或是刪除收藏影片
@router.get("/favorite_video")
def favorite_video( 
    order: str = Query(..., description="操作類型：add、delete、show"),
    collection_name:str = Query("最愛", description="收藏名稱，對於 add 操作是必填"),#預設是最愛
    video_id: int = Query(0, description="影片 ID，對於 show 操作可以是 0"),
    user_id: int = Depends(get_current_user)):

    if order.lower() not in ["add_collection",'add_collection_video', "delete_collection","delete_collection_video", "show_all_videos"]:
        raise HTTPException(status_code=400, detail="Invalid order parameter. Use 'add', 'delete', or 'show_collection' or 'show_collection_video'.")
    if order.lower() == "show_all_videos": 
        video_id = 0  # 如果是 show 操作，video_id 可以是 0
        collection_name = "最愛"
    else :
        video_id = int(video_id) # 確保 video_id 是整數
    
        
    conn = login_postgresql()
    cursor = conn.cursor()
    # 根據 order 參數決定操作
    switch = order.lower()
    match switch:
        case "delete_collection":
            cursor.execute("""
                select id
                from user_favorite_collection 
                where collection_name = %s and user_id = %s
                    """,(collection_name,user_id))
            collection_id =  cursor.fetchone()[0]
            # 刪除collection
            cursor.execute("""
                DELETE FROM user_favorite_collection
                WHERE id = %s 
            """, (collection_id,))
            cursor.execute("""
                delete from collection_video
                where collection_id = %s
            """,(collection_id,)
            )

        case "delete_collection_video":
            #檢查 video_id 是否有效
            if video_id <= 0:
                return {"message": "請提供有效的影片 ID"}
            # 檢查收藏記錄是否存在
            cursor.execute("""
                select 1 from collection_video cv
                join user_favorite_collection uc on cv.collection_id = uc.id
                WHERE uc.user_id = %s AND uc.collection_name = %s AND cv.video_id = %s
            """, (user_id, collection_name,video_id))
            if not cursor.fetchone():
                print("沒有找到要刪除的收藏記錄")
                return {"message": "沒有找到要刪除的收藏記錄"}  
            cursor.execute("""
                select id
                from user_favorite_collection 
                where collection_name = %s and user_id = %s
                    """,(collection_name,user_id))
            collection_id =  cursor.fetchone()[0]

            # 刪除收藏記錄
            cursor.execute("""
                DELETE FROM collection_video
                where collection_id = %s AND video_id = %s
            """, (collection_id,video_id))
            
            print("成功刪除收藏的影片")
        case "add_collection":
            # 新增收藏名稱
            if not collection_name:
                return {"message": "請提供有效的收藏名稱"}
            # 檢查是否已經存在同名收藏
            cursor.execute("""
                SELECT 1 FROM user_favorite_collection
                WHERE user_id = %s AND collection_name = %s
            """, (user_id, collection_name))
            if cursor.fetchone():
                return {"message": "收藏名稱已經存在"}
            # 新增收藏名稱
            cursor.execute("""
                INSERT INTO user_favorite_collection (user_id, collection_name)
                VALUES (%s, %s)
            """, (user_id, collection_name))
            if cursor.rowcount == 0:
                return {"message": "新增收藏失敗"}

        case "add_collection_video":
            # 新增收藏記錄
            if video_id <= 0:
                return {"message": "請提供有效的影片 ID"}
            
            # 檢查是否已經收藏過
            cursor.execute("""
                select 1 from collection_video cv
                join user_favorite_collection uc on cv.collection_id = uc.id
                WHERE uc.user_id = %s AND uc.collection_name = %s AND cv.video_id = %s
            """, (user_id, collection_name,video_id))
            if cursor.fetchone():
                return {"message": "影片已經收藏過了"}
            
            # 找到collection_id
            cursor.execute("""
                SELECT id FROM user_favorite_collection
                WHERE user_id = %s AND collection_name = %s
            """, (user_id, collection_name))
            collection_id = cursor.fetchone()
            # 根據collection_id新增收藏影片
            cursor.execute("""
                INSERT INTO collection_video (collection_id, video_id)
                VALUES (%s, %s)
            """, (collection_id[0], video_id))

            if cursor.rowcount == 0:
                return {"message": "新增收藏失敗"}
            
        # 一次抓出使用者的所有收藏集和對應影片資訊
        case "show_all_videos":
            cursor.execute("""
                SELECT 
                uc.collection_name,
                v.id AS video_id,
                v.title,
                v.embed_url,
                v.summary,
                v.description,
                COALESCE(array_remove(array_agg(DISTINCT t.name), NULL), '{}') AS tags
                FROM user_favorite_collection uc
                LEFT JOIN collection_video cv ON uc.id = cv.collection_id
                LEFT JOIN videos v ON cv.video_id = v.id
                LEFT JOIN LATERAL jsonb_array_elements_text(v.tag_ids) AS jt(tag_id_text) ON TRUE
                LEFT JOIN tags t ON t.id = jt.tag_id_text::int
                WHERE uc.user_id = %s
                GROUP BY uc.collection_name, v.id, v.title, v.embed_url, v.summary, v.description
                ORDER BY uc.collection_name, v.id;
            """, (user_id,))
            
            raw_data = cursor.fetchall()
            
            # 分組整理成 {collection_name: [影片們]}
            grouped = {}
            for row in raw_data:
                collection_name, video_id, title, embed_url, summary, description, tags = row
                video_obj = {
                    "video_id": video_id,
                    "title": title,
                    "embed_url": embed_url,
                    "summary": summary,
                    "description": description,
                    "tags": list(tags or [])  # 轉成 list，避免 None
                }
                grouped.setdefault(collection_name, []).append(video_obj)

            favorites = [
                {"collection_name": name, "videos": videos}
                for name, videos in grouped.items()
            ]

            return {"favorites": favorites}
    
    
    conn.commit()
    conn.close()
    
    return {"message": "success", "order": order, "video_id": video_id}

@router.get("/study_schedule")
def study_schedule(user_id: int = Depends(get_current_user)):
    conn = login_postgresql()
    cursor = conn.cursor()
    try:
        # 1) 取 map / phase 基本資料 + video_info
        cursor.execute("""
            SELECT
                lm.map_id,
                lm.query,
                lm.phase_number,
                lm.phase_title,
                lm.video_info
            FROM learning_map lm
            WHERE lm.user_id = %s
            ORDER BY lm.map_id ASC, lm.phase_number ASC, lm.id ASC;
        """, (user_id,))
        rows = cursor.fetchall()

        # 組 {map_id: {"query":..., "phases": {phase_number: {"phase_title":..., "videos": [dict...]}}}}
        maps: Dict[int, Dict[str, Any]] = {}

        for map_id, query, phase_number, phase_title, video_info in rows:
            m = maps.setdefault(int(map_id), {"query": str(query), "phases": {}})
            p = m["phases"].setdefault(int(phase_number), {
                "phase_title": str(phase_title or ""),
                "videos": []  # 暫存：每筆 item 最多 3 部
            })

            # 解析 video_info（可能是 jsonb->list 或 text）
            videos: List[dict] = []
            if isinstance(video_info, list):
                videos = video_info
            elif isinstance(video_info, str) and video_info.strip():
                try:
                    parsed = json.loads(video_info)
                    if isinstance(parsed, list):
                        videos = parsed
                except Exception:
                    videos = []

            # 這筆 item 取最多 3 部（你資料一筆就是 3 部）
            picked_this_item: List[Dict[str, Any]] = []
            for v in videos[:3]:
                if isinstance(v, dict) and v.get("video_id"):
                    picked_this_item.append({
                        "video_id": str(v.get("video_id")),  # 先存成字串，等會查 watched 會轉 int
                        "title": v.get("title", ""),
                        "url": v.get("url", "")
                    })
            p["videos"].extend(picked_this_item)

        # 2) Phase 層級：去重、最多保留 6 部；同時收集所有 video_id 以便一次查 watched
        all_video_ids_str: List[str] = []
        for m in maps.values():
            for ph in m["phases"].values():
                seen = set()
                dedup_videos: List[Dict[str, Any]] = []
                for v in ph["videos"]:
                    vid = v["video_id"]
                    if vid not in seen:
                        seen.add(vid)
                        dedup_videos.append(v)
                    if len(dedup_videos) >= 6:
                        break
                ph["videos"] = dedup_videos
                all_video_ids_str.extend([v["video_id"] for v in dedup_videos])

        # 3) 查 user_video_history：把要查的 video_id 轉成 int 陣列，避免 integer=text 衝突
        def to_int_list(ids: List[str]) -> List[int]:
            out: List[int] = []
            for s in ids:
                try:
                    out.append(int(s))
                except (TypeError, ValueError):
                    # 跳過非整數的 id（若你的 history.video_id 是 text，可以改為方案B：cast 比對）
                    pass
            return out

        watched_set: set[str] = set()
        ids_for_query: List[int] = to_int_list(all_video_ids_str)
        if ids_for_query:
            cursor.execute("""
                SELECT video_id
                FROM user_video_history
                WHERE user_id = %s AND video_id = ANY(%s)
            """, (user_id, ids_for_query))
            watched_rows = cursor.fetchall()
            # 轉回字串好跟 phase 裡的 video_id 對齊（那邊存的是字串）
            watched_set = {str(r[0]) for r in watched_rows}

        # 4) 標記 watched
        for m in maps.values():
            for ph in m["phases"].values():
                for v in ph["videos"]:
                    v["watched"] = (v["video_id"] in watched_set)

        # 5) 成績統計 & 錯題
        result: List[Dict[str, Any]] = []
        for map_id, info in maps.items():
            phases_out: List[Dict[str, Any]] = []
            for phase_number, data in sorted(info["phases"].items(), key=lambda x: x[0]):
                phase_title = data["phase_title"]
                videos = data["videos"]

                # 正確率
                cursor.execute("""
                    SELECT COUNT(*) AS total,
                           SUM(CASE WHEN correct THEN 1 ELSE 0 END) AS correct_count
                    FROM map_exam
                    WHERE user_id = %s AND map_id = %s AND phase_number = %s;
                """, (user_id, map_id, phase_number))
                total, correct_count = cursor.fetchone()
                accuracy = (float(correct_count or 0) / float(total)) if total and total > 0 else None

                # 錯題
                cursor.execute("""
                    SELECT question_number, question, options, answer, user_answer, explanation
                    FROM map_exam
                    WHERE user_id = %s AND map_id = %s AND phase_number = %s AND correct = FALSE
                    ORDER BY question_number ASC;
                """, (user_id, map_id, phase_number))
                wrong_rows = cursor.fetchall()
                if not total:
                    wrong_questions: Optional[List[Dict[str, Any]]] = None
                else:
                    wrong_questions = []
                    for qn, qtext, opts, ans, uans, explain in wrong_rows:
                        if isinstance(opts, str):
                            try:
                                opts = json.loads(opts)
                            except Exception:
                                opts = {"raw": opts}
                        wrong_questions.append({
                            "question_number": int(qn) if qn is not None else None,
                            "question": qtext,
                            "options": opts,
                            "answer": ans,
                            "user_answer": uans,
                            "explanation": explain
                        })

                phases_out.append({
                    "phase_number": int(phase_number),
                    "phase_title": phase_title,
                    "accuracy": accuracy,
                    "wrong_questions": wrong_questions,   # list 或 None
                    "videos": videos                      # [{video_id, title, url, watched}]
                })

            result.append({
                "map_id": int(map_id),
                "query": info["query"],
                "phases": phases_out
            })

        return result

    finally:
        try:
            cursor.close()
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass

#抓影片的連結，但先用不到
#串立收藏table
"""
create table user_favorite_video (
    user_id int not null,
    collection_name varchar(255) not null,
    video_id int not null,
    created_at timestamp not null default now(),
    primary key (user_id, collection_name, video_id),
"""
'''
@router.post("/yt-catch")
def yt_catch(keyword: str):
    videos = search_youtube_with_subtitles(keyword)
    conn = login_postgresql()
    for video in videos:
        download_and_save_to_postgresql(
            video_url=video["url"],
            title=video["title"],
            description=video.get("description", ""),
            conn=conn
        )
    conn.close()
    return {"keyword": keyword, "videos": videos}
'''
