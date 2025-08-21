import os
import chromadb
from chromadb.config import Settings

# 自動建立 chroma_db 目錄
db_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../chroma_db"))
os.makedirs(db_dir, exist_ok=True)

client = chromadb.Client(Settings(persist_directory=db_dir))

def query_similar_docs():
    return ("")