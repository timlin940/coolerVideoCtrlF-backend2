# 使用官方 Python 基礎映像
FROM python:3.10-slim

# 設定工作目錄
WORKDIR /app

# 複製 requirements.txt 並安裝依賴
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 複製專案所有檔案到容器
COPY . .

# 預設啟動 FastAPI 服務（可依你的主程式調整）
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]