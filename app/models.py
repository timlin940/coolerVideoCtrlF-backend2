from .db import Base
from sqlalchemy import Column, Integer, String, Text, JSON, TIMESTAMP, ForeignKey

class Video(Base):
    __tablename__ = "videos"
    id = Column(Integer, primary_key=True)
    url = Column(Text, unique=True, nullable=False)
    # ...其他欄位...

# 其他 model 也寫在這裡