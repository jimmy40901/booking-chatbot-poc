import sqlite3
from mcp.server.mcpserver import MCPServer
from init_db import DB_PATH
import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from google import genai

mcp = MCPServer("salon-tools")

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
faq_vectors = np.load(os.path.join(BASE_DIR, "data", "faq_vectors_q.npy"))
faq_df = pd.read_csv(os.path.join(BASE_DIR, "data", "faq.csv"), encoding="utf-8-sig")


def query_one(sql: str, params: tuple = ()):
    # 每次查詢都開新連線、用完就關（與 app.py 相同的寫法）
    conn = sqlite3.connect(DB_PATH)
    try:
        return conn.execute(sql, params).fetchone()
    finally:
        conn.close()

@mcp.tool()
def check_service_price(service: str) -> str:
    """查詢指定服務項目的價格"""
    row = query_one("SELECT price FROM services WHERE name = ?", (service,))
    return f"{row[0]}元" if row else "查無此服務項目"


@mcp.tool()
def check_opening_hours(day: str) -> str:
    """查詢指定星期幾的營業時間"""
    row = query_one("SELECT hours FROM business_hours WHERE weekday = ?", (day,))
    return row[0] if row else "查無此資料"


def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def retrieve(question: str) -> str:
    question_result = client.models.embed_content(model="gemini-embedding-001", contents=question)
    question_vector = question_result.embeddings[0].values

    best_score = -1
    best_index = 0
    for i in range(len(faq_vectors)):
        score = cosine_similarity(question_vector, faq_vectors[i])
        if score > best_score:
            best_score = score
            best_index = i

    # 用向量找出「第幾筆」，但交給 Gemini 的是那一筆完整的問題與回答
    row = faq_df.iloc[best_index]
    return f"問題：{row['question']}\n回答：{row['answer']}"


@mcp.tool()
def search_faq(question: str) -> str:
    """當客人詢問染髮後注意事項、燙髮保養、預約規則、染劑成分等非制式問題時，搜尋相關的服務說明文件"""
    return retrieve(question)


if __name__ == "__main__":
    mcp.run()