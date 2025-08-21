from fastapi import FastAPI
from fastapi import Query
from chroma.vector_store import query_similar_docs #,add_doc
from fastapi.middleware.cors import CORSMiddleware
#from chroma.faq_seed import seed_faq

# 修改導入方式
from app.config import settings
from app.api import video_router, chroma_router


# 創建 FastAPI 實例
app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION,
    debug=settings.DEBUG
)

app.add_middleware(
    CORSMiddleware,
    # allow_origins=["*"],  # 開發階段先全部開放
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 註冊路由
app.include_router(video_router)
app.include_router(chroma_router)

@app.get("/")
async def root():
    return {"message" : "Hello Functrol!!"}


# test ===================================

# 查詢 FAQ
@app.get("/search/")
def search_faq(q: str = Query(..., description="使用者查詢問題")):
    result = query_similar_docs(q)
    return {"results": result}


# main.py

print("Seeding FAQ...")
#seed_faq()
print("Seeding done.")


# 新增 JWT 認證支援，用來處理videos_route的get_current_user(request: Request)，在docs可以輸入token測試
from fastapi.openapi.models import APIKey, APIKeyIn, SecuritySchemeType
from fastapi.openapi.utils import get_openapi

from fastapi.security import HTTPBearer

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="Video Search API",
        version="1.0",
        description="基於 FastAPI 的影片搜尋 API",
        routes=app.routes,
    )
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
        }
    }
    for path in openapi_schema["paths"].values():
        for method in path.values():
            method["security"] = [{"BearerAuth": []}]
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi