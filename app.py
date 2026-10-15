import gradio as gr

try:
    import spaces

    @spaces.GPU
    def _warmup():
        return "ok"
except ImportError:
    pass
import os
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from google import genai
from google.genai import types
from dataclasses import dataclass
from typing import TypedDict
from langgraph.graph import StateGraph, END
import numpy as np
import pandas as pd
from init_db import init_db, DB_PATH

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

# ====================================================
# 區塊 1：客戶資料模板
# ====================================================

@dataclass
class Customer:
    name: str
    is_new: bool
    visit_count: int = 0
    is_birthday_month: bool = False

# ====================================================
# 區塊 2：FAQ Agent 所需的一切(Phase 1-2：Tool Calling + RAG)
# ====================================================

# ---------- 資料庫：營業時間、價格、客戶、預約都存在 SQLite(由 init_db.py 建立) ----------
init_db()  # 資料庫不存在、或還是空的時候，自動建立並塞入初始資料

def query_one(sql: str, params: tuple = ()):
    # 每次查詢都開一條新連線、用完就關：SQLite 的連線不能跨執行緒共用，
    # 而 Gradio 與 LangGraph 可能在不同的執行緒上執行
    conn = sqlite3.connect(DB_PATH)
    try:
        return conn.execute(sql, params).fetchone()
    finally:
        conn.close()

def check_opening_hours(day: str) -> str:
    """查詢指定星期幾的營業時間"""
    row = query_one("SELECT hours FROM business_hours WHERE weekday = ?", (day,))
    return row[0] if row else "查無此資料"

def check_service_price(service: str) -> str:
    """查詢指定服務項目的價格"""
    row = query_one("SELECT price FROM services WHERE name = ?", (service,))
    return f"{row[0]}元" if row else "查無此服務項目"

# ---------- FAQ 資料：讀取 CSV 與預先建好的向量(由 build_index.py 產生) ----------
# 用「這個檔案所在的資料夾」當基準，不管從哪裡執行都找得到 data 資料夾
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

faq_df = pd.read_csv(os.path.join(BASE_DIR, "data", "faq.csv"), encoding="utf-8-sig")
faq_vectors = np.load(os.path.join(BASE_DIR, "data", "faq_vectors_q.npy"))  # 第 i 列對應 faq_df 的第 i 筆

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

def search_faq(question: str) -> str:
    """當客人詢問染髮後注意事項、燙髮保養、預約規則、染劑成分等非制式問題時，搜尋相關的服務說明文件"""
    return retrieve(question)

tools = types.Tool(function_declarations=[
    {
        "name": "check_opening_hours",
        "description": "查詢指定星期幾的營業時間",
        "parameters": {"type": "object", "properties": {
            "day": {"type": "string", "description": "星期幾，例如：禮拜一、禮拜二"}
        }, "required": ["day"]}
    },
    {
        "name": "check_service_price",
        "description": "查詢指定服務項目的價格",
        "parameters": {"type": "object", "properties": {
            "service": {"type": "string", "description": "服務項目名稱，例如：剪髮、染髮"}
        }, "required": ["service"]}
    },
    {
        "name": "search_faq",
        "description": "當客人詢問染髮後注意事項、燙髮保養、預約規則、染劑成分等非制式問題時，搜尋相關的服務說明文件",
        "parameters": {"type": "object", "properties": {
            "question": {"type": "string", "description": "客人詢問的問題原文"}
        }, "required": ["question"]}
    }
])

def build_system_prompt(customer: Customer) -> str:
    base = "你是一間髮廊的客服人員，語氣親切。請回答營業時間、服務價格、服務內容、預約規則與髮質保養相關的問題。若查詢不到相關資訊，請誠實告知無法確定，不要編造答案。回答時只根據查詢到的內容，不要自行補充查詢結果沒有提到的建議或規定。"
    if customer.is_new:
        extra = "這位是新客，請特別介紹一下我們的服務，並歡迎他第一次光臨。"
    elif customer.visit_count >= 5:
        extra = f"這位是熟客（已經來訪 {customer.visit_count} 次），可以用更輕鬆熟悉的語氣對話，並主動提及老客戶有 9 折優惠。"
    else:
        extra = "這位是一般舊客，正常親切應對即可。"
    return base + extra

def ask(question: str, customer: Customer) -> str:
    system_prompt = build_system_prompt(customer)
    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=question,
        config=types.GenerateContentConfig(tools=[tools], system_instruction=system_prompt)
    )
    part = response.candidates[0].content.parts[0]

    if part.function_call is None:
        return response.text

    function_call = part.function_call
    if function_call.name == "check_opening_hours":
        tool_result = check_opening_hours(**function_call.args)
    elif function_call.name == "check_service_price":
        tool_result = check_service_price(**function_call.args)
    elif function_call.name == "search_faq":
        tool_result = search_faq(**function_call.args)

    final_response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=[
            question,
            response.candidates[0].content,
            types.Content(role="user", parts=[types.Part.from_function_response(
                name=function_call.name, response={"result": tool_result}
            )])
        ],
        config=types.GenerateContentConfig(tools=[tools], system_instruction=system_prompt)
    )
    return final_response.text

# ====================================================
# 區塊 3：Booking Agent 所需的一切(Phase 3：LangGraph 子圖)
# ====================================================

WEEKDAYS = ["禮拜一", "禮拜二", "禮拜三", "禮拜四", "禮拜五", "禮拜六", "禮拜天"]
TAIPEI = timezone(timedelta(hours=8))  # 台灣時區(UTC+8)，避免伺服器在 UTC 時算錯星期

def today_info() -> dict:
    now = datetime.now(TAIPEI)
    wd = now.weekday()  # 星期一是 0，星期天是 6
    return {
        "date": now.strftime("%Y-%m-%d"),
        "today": WEEKDAYS[wd],
        "tomorrow": WEEKDAYS[(wd + 1) % 7],
        "day_after": WEEKDAYS[(wd + 2) % 7],
    }

def extract_booking_info(user_message: str) -> dict:
    t = today_info()
    prompt = f"""從客人的話中提取預約資訊，以 JSON 回覆，格式如下：
{{"service": "...", "day": "..."}}

今天是 {t['today']}（{t['date']}），明天是 {t['tomorrow']}，後天是 {t['day_after']}。

service 只能是：剪髮、洗髮、染髮、燙髮。判斷不出來填「不明確」。
day 只能是：禮拜一、禮拜二、禮拜三、禮拜四、禮拜五、禮拜六、禮拜天。
客人說「今天、明天、後天」時，請依上面的對照換算成星期；客人完全沒提到日期，填「未指定」。

客人的話：{user_message}
"""
    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json")
    )
    try:
        info = json.loads(response.text)
    except json.JSONDecodeError:
        info = {}
    return {
        "service": info.get("service", "不明確"),
        "day": info.get("day", "未指定"),
    }

class BookingState(TypedDict):
    user_message: str
    service: str
    day: str
    is_available: bool

def extract_info(state: BookingState) -> dict:
    info = extract_booking_info(state['user_message'])
    # 這句話有明確說的就用新的；沒說的，沿用上一輪記住的(state 裡原本的值)
    service = info['service'] if info['service'] != "不明確" else (state['service'] or "不明確")
    day = info['day'] if info['day'] != "未指定" else (state['day'] or "未指定")
    print(f"判斷出的預約資訊：服務={service}，日期={day}")
    return {"service": service, "day": day}

def ask_clarification(state: BookingState) -> dict:
    print("不好意思，請問您想預約剪髮、洗髮、染髮還是燙髮呢？")
    return {}

def ask_day(state: BookingState) -> dict:
    print("請問您想預約哪一天呢？")
    return {}

def check_availability(state: BookingState) -> dict:
    print(f"查詢 {state['day']} 是否有空位...")
    # 營業時間只有 check_opening_hours 一個來源，不另外維護第二份名單
    hours = check_opening_hours(state['day'])
    available = hours not in ("公休", "查無此資料")
    return {"is_available": available}

def confirm_booking(state: BookingState) -> dict:
    print(f"已為您確認預約：{state['service']}，{state['day']}")
    return {}

def suggest_reschedule(state: BookingState) -> dict:
    print(f"很抱歉，{state['day']} 公休，建議您改約其他時間")
    return {}

def route_after_extract(state: BookingState) -> str:
    if state['service'] == "不明確":
        return "ask_clarification"
    if state['day'] == "未指定":
        return "ask_day"
    return "check_availability"

def route_after_check(state: BookingState) -> str:
    return "confirm_booking" if state['is_available'] else "suggest_reschedule"

booking_graph = StateGraph(BookingState)
booking_graph.add_node("extract_info", extract_info)
booking_graph.add_node("ask_clarification", ask_clarification)
booking_graph.add_node("ask_day", ask_day)
booking_graph.add_node("check_availability", check_availability)
booking_graph.add_node("confirm_booking", confirm_booking)
booking_graph.add_node("suggest_reschedule", suggest_reschedule)
booking_graph.set_entry_point("extract_info")
booking_graph.add_conditional_edges("extract_info", route_after_extract, {
    "ask_clarification": "ask_clarification",
    "ask_day": "ask_day",
    "check_availability": "check_availability"
})
booking_graph.add_edge("ask_clarification", END)
booking_graph.add_edge("ask_day", END)
booking_graph.add_conditional_edges("check_availability", route_after_check, {
    "confirm_booking": "confirm_booking", "suggest_reschedule": "suggest_reschedule"
})
booking_graph.add_edge("confirm_booking", END)
booking_graph.add_edge("suggest_reschedule", END)

booking_app = booking_graph.compile()

def booking_reply(result: dict) -> str:
    """把預約子圖跑完的最終 state，轉成要回給客人的一句話"""
    if result['service'] == "不明確":
        if result['day'] != "未指定":
            return f"好的，{result['day']}。請問您想預約剪髮、洗髮、染髮還是燙髮呢？"
        return "不好意思，請問您想預約剪髮、洗髮、染髮還是燙髮呢？"
    if result['day'] == "未指定":
        return f"好的，{result['service']}沒問題。請問您想預約哪一天呢？"
    if result['is_available']:
        return f"已為您確認預約：{result['service']}，{result['day']}"
    return f"很抱歉，{result['day']} 公休，建議您改約其他時間"

# ====================================================
# 區塊 4：最外層的蜂巢式多 Agent 大圖
# ====================================================

class HiveState(TypedDict):
    user_message: str
    intent: str
    service: str
    day: str
    is_available: bool
    final_response: str
    draft: dict  # 這位客人「預約到一半」的資訊，例如 {"service": "剪髮", "day": ""}

def describe_draft(draft: dict) -> str:
    parts = []
    if draft.get("service"):
        parts.append(f"服務：{draft['service']}")
    if draft.get("day"):
        parts.append(f"日期：{draft['day']}")
    return "、".join(parts) if parts else "（尚無）"

def router(state: HiveState) -> dict:
    user_input = state['user_message']
    draft = state.get('draft') or {}
    pending = ""
    if draft.get("service") or draft.get("day"):
        pending = (
            f"\n目前客人正在進行預約，系統已記住：{describe_draft(draft)}。"
            "如果客人這句話是在補充或修改預約資訊（例如回答服務項目或日期），請回 booking；"
            "如果是在問其他問題，請回 faq。\n"
        )
    prompt = f"""判斷客人這句話的意圖，只能回答以下兩種之一：
- booking：如果客人想要預約、約時間
- faq：如果客人是在詢問問題（價格、營業時間、注意事項等）
{pending}
只回傳 booking 或 faq，不要有其他文字。

客人的話：{user_input}
"""
    response = client.models.generate_content(model="gemini-3.5-flash-lite", contents=prompt)
    intent = response.text.strip()
    print(f"判斷意圖：{intent}")
    return {"intent": intent}

def faq_agent(state: HiveState) -> dict:
    user_input = state['user_message']
    dummy_customer = Customer(name="訪客", is_new=False, visit_count=2)
    response_text = ask(user_input, dummy_customer)
    print(f"FAQ Agent 回覆：{response_text}")
    return {"final_response": response_text}

def next_draft(result: dict) -> dict:
    """依預約子圖的結果，決定下一輪要記住什麼"""
    if result['service'] != "不明確" and result['day'] != "未指定" and result['is_available']:
        return {}  # 預約完成，清空記憶
    service = "" if result['service'] == "不明確" else result['service']
    day = "" if result['day'] == "未指定" else result['day']
    if service and day and not result['is_available']:
        day = ""  # 那天不能約：保留服務，等客人改別天
    return {"service": service, "day": day}

def booking_agent(state: HiveState) -> dict:
    draft = state.get('draft') or {}
    booking_initial_state = {
        "user_message": state['user_message'],
        "service": draft.get("service", ""),
        "day": draft.get("day", ""),
        "is_available": False
    }
    booking_result = booking_app.invoke(booking_initial_state)
    return {
        "final_response": booking_reply(booking_result),
        "draft": next_draft(booking_result),
    }

def route_after_router(state: HiveState) -> str:
    return "booking_agent" if state['intent'] == "booking" else "faq_agent"

hive_graph = StateGraph(HiveState)
hive_graph.add_node("router", router)
hive_graph.add_node("faq_agent", faq_agent)
hive_graph.add_node("booking_agent", booking_agent)
hive_graph.set_entry_point("router")
hive_graph.add_conditional_edges("router", route_after_router, {
    "faq_agent": "faq_agent", "booking_agent": "booking_agent"
})
hive_graph.add_edge("faq_agent", END)
hive_graph.add_edge("booking_agent", END)

hive_app = hive_graph.compile()

# ---------- Gradio 介面 ----------

def chatbot_interface(question: str, draft: dict):
    draft = draft or {}
    result = hive_app.invoke({
        "user_message": question,
        "intent": "",
        "service": "",
        "day": "",
        "is_available": False,
        "final_response": "",
        "draft": draft
    })
    new_draft = result.get("draft") or {}
    return result['final_response'], describe_draft(new_draft), new_draft

demo = gr.Interface(
    fn=chatbot_interface,
    inputs=[
        gr.Textbox(label="請輸入您的問題（可以問價格/營業時間/保養問題，也可以直接說想預約）"),
        gr.State({}),
    ],
    outputs=[
        gr.Textbox(label="客服回覆"),
        gr.Textbox(label="系統目前記住的預約資訊"),
        gr.State(),
    ],
    title="髮廊客服 Multi-Agent Chatbot",
    description="蜂巢式多 Agent 架構：Router 自動判斷意圖，分派給 FAQ Agent（Tool Calling + RAG）或 Booking Agent（LangGraph 引導式流程）。預約資訊不完整時會追問，可在同一頁連續輸入；重新整理頁面會重置記憶。"
)

demo.launch()