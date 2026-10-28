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
import time
import secrets
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
from mcp_client import call_mcp_tool

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
    id: int | None = None  # 資料庫裡的編號；尚未存進資料庫的客人是 None

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

# ---------- 辨識客人：用姓名到資料庫查詢(之後接 LINE 時，改成用 LINE 的使用者 ID 查) ----------

def find_customer(name: str):
    row = query_one("SELECT id, name, visit_count FROM customers WHERE name = ?", (name,))
    if row is None:
        return None
    customer_id, customer_name, visit_count = row
    return Customer(name=customer_name, is_new=(visit_count == 0), visit_count=visit_count, id=customer_id)

def identify_customer(name: str) -> Customer:
    name = (name or "").strip()
    if not name:
        return Customer(name="訪客", is_new=False, visit_count=0)  # 沒提供姓名：一般客人
    found = find_customer(name)
    if found:
        return found
    return Customer(name=name, is_new=True, visit_count=0)  # 資料庫查無此人：當作新客(尚未寫入資料庫)

MAX_PER_DAY = 5  # 每天最多接幾位客人(同一位客人同一天預約多項服務，視為同一次到店，只佔一個名額)

def query_all(sql: str, params: tuple = ()):
    conn = sqlite3.connect(DB_PATH)
    try:
        return conn.execute(sql, params).fetchall()
    finally:
        conn.close()

def count_visits(date_str: str) -> int:
    """這一天有幾位客人預約(同一位客人預約多項服務，只算一位)"""
    row = query_one(
        "SELECT COUNT(DISTINCT customer_id) FROM appointments WHERE date = ? AND status = 'confirmed'",
        (date_str,)
    )
    return row[0]

def booked_services(customer_name: str, date_str: str) -> list:
    """這位客人這一天已經預約了哪些服務，回傳 [(預約編號, 服務), ...]"""
    name = (customer_name or "").strip()[:30]
    if not name:
        return []
    return query_all(
        """SELECT a.id, a.service FROM appointments a JOIN customers c ON c.id = a.customer_id
           WHERE c.name = ? AND a.date = ? AND a.status = 'confirmed' ORDER BY a.id""",
        (name, date_str)
    )

def create_appointment(customer_name: str, service: str, date_str: str) -> int:
    """寫入一筆預約，回傳預約編號；不能寫入(額滿，或同一天同一項服務已預約過)時回傳 0。
    客人不在資料庫時，先建立客人(新客，來訪 0 次)。"""
    name = customer_name.strip()[:30]
    conn = sqlite3.connect(DB_PATH)
    try:
        row = conn.execute("SELECT id FROM customers WHERE name = ?", (name,)).fetchone()
        if row is None:
            cur = conn.execute("INSERT INTO customers (name, visit_count) VALUES (?, 0)", (name,))
            customer_id = cur.lastrowid
        else:
            customer_id = row[0]

        # 「檢查」和「寫入」放在同一個 SQL 動作裡，就算兩個請求幾乎同時進來也不會出錯：
        #  (1) 同一位客人、同一天、同一項服務還沒預約過
        #  (2) 這位客人這天已經有預約(加一項服務算同一次到店)，或是當天客人數還沒滿
        cur = conn.execute(
            """INSERT INTO appointments (customer_id, service, date)
               SELECT ?, ?, ?
               WHERE NOT EXISTS (SELECT 1 FROM appointments
                                 WHERE customer_id = ? AND date = ? AND service = ? AND status = 'confirmed')
                 AND (EXISTS (SELECT 1 FROM appointments
                              WHERE customer_id = ? AND date = ? AND status = 'confirmed')
                      OR (SELECT COUNT(DISTINCT customer_id) FROM appointments
                          WHERE date = ? AND status = 'confirmed') < ?)""",
            (customer_id, service, date_str,
             customer_id, date_str, service,
             customer_id, date_str,
             date_str, MAX_PER_DAY)
        )
        if cur.rowcount == 0:
            conn.rollback()  # 沒寫入：連剛剛新建的客人一起復原
            return 0
        conn.commit()  # 要 commit，寫入才會真的存進檔案
        return cur.lastrowid
    finally:
        conn.close()

# ---------- 對話紀錄：記下每一輪「客人問了什麼、系統怎麼答」 ----------

def log_conversation(session_id: str, customer_name: str, user_message: str,
                     intent: str, reply: str, latency_ms: int) -> None:
    """記錄一輪對話。記錄失敗不能影響客人使用，所以任何錯誤都只印出來，不往外丟。
    輸入與回覆都限制長度，避免被塞入超長內容。"""
    try:
        now = datetime.now(TAIPEI).strftime("%Y-%m-%d %H:%M:%S")
        conn = sqlite3.connect(DB_PATH)
        try:
            conn.execute(
                """INSERT INTO conversations
                   (created_at, session_id, customer_name, user_message, intent, reply, latency_ms)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (now, session_id or "", (customer_name or "")[:30], (user_message or "")[:500],
                 intent or "", (reply or "")[:1000], latency_ms)
            )
            conn.commit()
        finally:
            conn.close()
    except Exception as e:
        print(f"[log error] {type(e).__name__}")  # 只印錯誤種類，不印使用者輸入

ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")  # 管理後台密碼；沒有設定就不開放

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
    if customer.name != "訪客":
        extra += f"客人的名字是{customer.name}，可以適度稱呼對方。"
    return base + extra

ALLOWED_TOOLS = {"check_opening_hours", "check_service_price", "search_faq"}
def ask(question: str, customer: Customer) -> str:
    system_prompt = build_system_prompt(customer)
    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=question,
        config=types.GenerateContentConfig(tools=[tools], system_instruction=system_prompt)
    )
    parts = response.candidates[0].content.parts
    function_calls = [p.function_call for p in parts if p.function_call]

    if not function_calls:
        return response.text

    response_parts = []
    for function_call in function_calls:
        if function_call.name not in ALLOWED_TOOLS:
            return "抱歉，我暫時無法處理這個問題。"
        try:
            tool_result = call_mcp_tool(function_call.name, dict(function_call.args))
        except Exception as e:
            print("MCP 工具呼叫失敗：", type(e).__name__)
            return "抱歉，系統暫時無法查詢，請稍後再試。"
        response_parts.append(types.Part.from_function_response(
            name=function_call.name, response={"result": tool_result}
        ))

    final_response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=[
            question,
            response.candidates[0].content,
            types.Content(role="user", parts=response_parts)
        ],
        config=types.GenerateContentConfig(tools=[tools], system_instruction=system_prompt)
    )
    return final_response.text

# ====================================================
# 區塊 3：Booking Agent 所需的一切(Phase 3：LangGraph 子圖)
# ====================================================

WEEKDAYS = ["禮拜一", "禮拜二", "禮拜三", "禮拜四", "禮拜五", "禮拜六", "禮拜天"]
TAIPEI = timezone(timedelta(hours=8))  # 台灣時區(UTC+8)，避免伺服器在 UTC 時算錯日期
VALID_DAYS = ["今天", "明天", "後天"] + WEEKDAYS + ["下" + d for d in WEEKDAYS]
VALID_SERVICES = ["剪髮", "洗髮", "染髮", "燙髮"] #固定的服務
SERVICE_SYNONYMS = {  #相似的歸類成我們上面要的固定的格式
    "理髮": "剪髮",
    "剪頭髮": "剪髮",
    "理頭髮": "剪髮",
    "洗頭": "洗髮",
    "洗頭髮": "洗髮",
    "染頭髮": "染髮",
    "燙頭髮": "燙髮",
}

def resolve_date(day_phrase: str, today=None) -> str:
    """把客人說的日期說法換算成真實日期(YYYY-MM-DD)；無法判斷時回傳空字串。
    規則：「今天、明天、後天」照字面；「禮拜X」是從明天起算最近的那一天
    (客人說的星期剛好是今天的星期時，視為下週；要指今天，客人會說「今天」)。"""
    if today is None:
        today = datetime.now(TAIPEI).date()
    if day_phrase == "今天":
        target = today
    elif day_phrase == "明天":
        target = today + timedelta(days=1)
    elif day_phrase == "後天":
        target = today + timedelta(days=2)
    elif day_phrase in WEEKDAYS:
        delta = (WEEKDAYS.index(day_phrase) - today.weekday()) % 7  # 星期一是 0，星期天是 6
        if delta == 0:
            delta = 7
        target = today + timedelta(days=delta)
    elif day_phrase.startswith("下") and day_phrase[1:] in WEEKDAYS:
        next_monday = today + timedelta(days=7 - today.weekday())
        target = next_monday + timedelta(days=WEEKDAYS.index(day_phrase[1:]))
    else:
        return ""
    return target.isoformat()

def weekday_label(date_str: str) -> str:
    return WEEKDAYS[datetime.strptime(date_str, "%Y-%m-%d").weekday()]

def fmt_date(date_str: str) -> str:
    return f"{date_str}（{weekday_label(date_str)}）"

def find_open_dates(start_date_str: str, days: int = 14, limit: int = 3) -> list:
    """從指定日期的隔天起算，往後 days 天內，找出「有營業、而且還沒額滿」的日子"""
    start = datetime.strptime(start_date_str, "%Y-%m-%d").date()
    found = []
    for i in range(1, days + 1):
        d = start + timedelta(days=i)
        d_str = d.isoformat()
        hours = check_opening_hours(WEEKDAYS[d.weekday()])
        if hours in ("公休", "查無此資料"):
            continue
        if count_visits(d_str) >= MAX_PER_DAY:
            continue
        found.append(d_str)
        if len(found) == limit:
            break
    return found

def extract_booking_info(user_message: str) -> dict:
    """請 LLM 只負責「聽懂客人的說法」；換算成真實日期交給 Python(resolve_date)。"""
    prompt = f"""從客人的話中提取預約資訊，以 JSON 回覆，格式如下：
{{"service": "...", "day": "..."}}

day 只能是：今天、明天、後天、禮拜一、禮拜二、禮拜三、禮拜四、禮拜五、禮拜六、禮拜天，
或「下禮拜」加上星期幾（下禮拜一、下禮拜二、下禮拜三、下禮拜四、下禮拜五、下禮拜六、下禮拜天）。
「星期」、「週」、「禮拜」是同一個意思，請統一寫成「禮拜」，例如客人說「下星期四」就填「下禮拜四」。
只要把說法寫成上面的選項，不要自己計算日期；客人沒提到日期，或說法無法對應（例如下個月、大後天），填「未指定」。

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
    service = info.get("service", "不明確")
    service = SERVICE_SYNONYMS.get(service, service)
    day = info.get("day", "未指定")
    return {
        "service": service if service in VALID_SERVICES else "不明確",
        "day": day if day in VALID_DAYS else "未指定",
    }

class BookingState(TypedDict):
    user_message: str
    customer_name: str
    service: str         # 剪髮 / 洗髮 / 染髮 / 燙髮，尚未決定為「不明確」
    date: str            # 真實日期 YYYY-MM-DD，尚未決定為空字串
    is_available: bool
    appointment_id: int  # 預約成功後的預約編號，未成功為 0
    reason: str          # 不能預約的原因：closed(公休) / full(額滿) / duplicate(重複預約)，可以預約則為空字串
    suggestions: list    # 不能預約時，主動建議的其他日期
    existing: str        # 重複預約時，說明已經存在的那一筆
    same_visit: list     # 這位客人這天已經預約的其他服務(同一次到店一起進行)

def extract_info(state: BookingState) -> dict:
    info = extract_booking_info(state['user_message'])
    # 這句話有明確說的就用新的；沒說的，沿用上一輪記住的
    service = info['service'] if info['service'] != "不明確" else (state['service'] or "不明確")
    date = resolve_date(info['day']) or state['date']
    print(f"判斷出的預約資訊：服務={service}，日期={date or '未指定'}")
    return {"service": service, "date": date}

def ask_clarification(state: BookingState) -> dict:
    print("不好意思，請問您想預約剪髮、洗髮、染髮還是燙髮呢？")
    return {}

def ask_day(state: BookingState) -> dict:
    print("請問您想預約哪一天呢？")
    return {}

def check_availability(state: BookingState) -> dict:
    date = state['date']
    print(f"查詢 {date} 是否有空位...")
    # 營業時間只有 check_opening_hours 一個來源，不另外維護第二份名單
    hours = check_opening_hours(weekday_label(date))
    if hours in ("公休", "查無此資料"):
        return {"is_available": False, "reason": "closed", "suggestions": find_open_dates(date), "same_visit": []}

    mine = booked_services(state['customer_name'], date)  # 這位客人這天已預約的 [(編號, 服務), ...]
    for appointment_id, service in mine:
        if service == state['service']:
            return {"is_available": False, "reason": "duplicate", "existing": f"預約編號 {appointment_id}", "same_visit": []}

    # 這位客人這天已經有預約時，再加一項服務算「同一次到店」，不佔新的名額；否則要看名額有沒有滿
    if not mine and count_visits(date) >= MAX_PER_DAY:
        return {"is_available": False, "reason": "full", "suggestions": find_open_dates(date), "same_visit": []}

    return {"is_available": True, "reason": "", "suggestions": [], "existing": "",
            "same_visit": [service for _, service in mine]}

def ask_name(state: BookingState) -> dict:
    print("需要姓名才能完成預約")
    return {}

def confirm_booking(state: BookingState) -> dict:
    appointment_id = create_appointment(state['customer_name'], state['service'], state['date'])
    if appointment_id == 0:
        # 剛才檢查時可以預約，寫入時卻不行(剛好被別人訂滿，或重複送出)：重新判斷一次原因
        print(f"寫入失敗：{state['date']}，重新判斷原因")
        recheck = check_availability(state)
        if recheck["is_available"]:
            recheck = {"is_available": False, "reason": "full", "suggestions": find_open_dates(state['date'])}
        return recheck
    print(f"已寫入預約 #{appointment_id}：{state['customer_name']}，{state['service']}，{state['date']}")
    return {"appointment_id": appointment_id}

def suggest_reschedule(state: BookingState) -> dict:
    print(f"{state['date']} 無法預約(原因：{state['reason']})，建議改約：{state['suggestions']}")
    return {}

def route_after_extract(state: BookingState) -> str:
    if state['service'] == "不明確":
        return "ask_clarification"
    if not state['date']:
        return "ask_day"
    return "check_availability"

def route_after_check(state: BookingState) -> str:
    if not state['is_available']:
        return "suggest_reschedule"
    if not state['customer_name'].strip():
        return "ask_name"
    return "confirm_booking"

booking_graph = StateGraph(BookingState)
booking_graph.add_node("extract_info", extract_info)
booking_graph.add_node("ask_clarification", ask_clarification)
booking_graph.add_node("ask_day", ask_day)
booking_graph.add_node("check_availability", check_availability)
booking_graph.add_node("ask_name", ask_name)
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
    "suggest_reschedule": "suggest_reschedule",
    "ask_name": "ask_name",
    "confirm_booking": "confirm_booking"
})
booking_graph.add_edge("ask_name", END)
booking_graph.add_edge("confirm_booking", END)
booking_graph.add_edge("suggest_reschedule", END)

booking_app = booking_graph.compile()

def booking_reply(result: dict) -> str:
    """把預約子圖跑完的最終 state，轉成要回給客人的一句話"""
    if result['service'] == "不明確":
        if result['date']:
            return f"好的，{fmt_date(result['date'])}。請問您想預約剪髮、洗髮、染髮還是燙髮呢？"
        return "不好意思，請問您想預約剪髮、洗髮、染髮還是燙髮呢？"
    if not result['date']:
        return f"好的，{result['service']}沒問題。請問您想預約哪一天呢？"
    if not result['is_available']:
        d = fmt_date(result['date'])
        if result['reason'] == "duplicate":
            return f"您在{d}已經預約了{result['service']}（{result['existing']}），不需要重複預約喔。"
        why = "公休" if result['reason'] == "closed" else "預約已額滿"
        if result['suggestions']:
            options = "、".join(fmt_date(x) for x in result['suggestions'])
            return f"很抱歉，{d}{why}。最近還有空位的日子：{options}。請問您想改約哪一天呢？"
        return f"很抱歉，{d}{why}，而且接下來兩週都沒有空位，建議您稍後再來詢問。"
    if not result['appointment_id']:
        return f"{fmt_date(result['date'])}有空位。請在上方「您的姓名」欄位填入姓名後，回覆「確認」即可完成預約。"
    msg = f"已為您確認預約：{result['service']}，{fmt_date(result['date'])}，預約編號 {result['appointment_id']}。"
    if result['same_visit']:
        msg += f"您這天另外預約了{'、'.join(result['same_visit'])}，會安排在同一次到店一起進行。"
    return msg

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
    customer_name: str  # 客人在介面上輸入的姓名(可空白)
    draft: dict  # 這位客人「預約到一半」的資訊，例如 {"service": "剪髮", "date": ""}

def describe_draft(draft: dict) -> str:
    parts = []
    if draft.get("service"):
        parts.append(f"服務：{draft['service']}")
    if draft.get("date"):
        parts.append(f"日期：{fmt_date(draft['date'])}")
    return "、".join(parts) if parts else "（尚無）"

def router(state: HiveState) -> dict:
    user_input = state['user_message']
    draft = state.get('draft') or {}
    pending = ""
    if draft.get("service") or draft.get("date"):
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
    customer = identify_customer(state.get('customer_name', ''))
    response_text = ask(user_input, customer)
    print(f"FAQ Agent 回覆：{response_text}")
    return {"final_response": response_text}

def next_draft(result: dict) -> dict:
    """依預約子圖的結果，決定下一輪要記住什麼"""
    if result['appointment_id'] or result['reason'] == "duplicate":
        return {}  # 預約完成，或這天已經有預約：清空記憶
    service = "" if result['service'] == "不明確" else result['service']
    date = result['date']
    if result['reason'] in ("closed", "full"):
        date = ""  # 那天不能約：保留服務，等客人改別天
    return {"service": service, "date": date}

def booking_agent(state: HiveState) -> dict:
    draft = state.get('draft') or {}
    booking_initial_state = {
        "user_message": state['user_message'],
        "customer_name": (state.get('customer_name') or "").strip(),
        "service": draft.get("service", ""),
        "date": draft.get("date", ""),
        "is_available": False,
        "appointment_id": 0,
        "reason": "",
        "suggestions": [],
        "existing": "",
        "same_visit": [],
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

def chatbot_interface(question: str, customer_name: str, draft: dict, request: gr.Request = None):
    draft = draft or {}
    start = time.perf_counter()
    result = hive_app.invoke({
        "user_message": question,
        "intent": "",
        "service": "",
        "day": "",
        "is_available": False,
        "final_response": "",
        "customer_name": customer_name or "",
        "draft": draft
    })
    latency_ms = int((time.perf_counter() - start) * 1000)
    new_draft = result.get("draft") or {}

    session_id = getattr(request, "session_hash", "") or ""
    log_conversation(session_id, customer_name, question, result.get("intent", ""),
                     result['final_response'], latency_ms)
    return result['final_response'], describe_draft(new_draft), new_draft

demo = gr.Interface(
    fn=chatbot_interface,
    inputs=[
        gr.Textbox(label="請輸入您的問題（可以問價格/營業時間/保養問題，也可以直接說想預約）"),
        gr.Textbox(label="您的姓名（選填，系統會查詢會員資料。可試試：小華、小明、小美）"),
        gr.State({}),
    ],
    outputs=[
        gr.Textbox(label="客服回覆"),
        gr.Textbox(label="系統目前記住的預約資訊"),
        gr.State(),
    ],
    title="髮廊客服 Multi-Agent Chatbot",
    description=(
        "蜂巢式多 Agent 架構：Router 自動判斷意圖，分派給 FAQ Agent（Tool Calling + RAG）"
        "或 Booking Agent（LangGraph 引導式流程）。預約資訊不完整時會追問，可在同一頁連續輸入；"
        "重新整理頁面會重置記憶。\n\n"
        "⚠️ 為了改善系統，您輸入的內容與系統回覆會被記錄。請勿輸入真實個資，姓名欄請使用暱稱。"
    )
)

# ---------- 管理後台：用密碼保護，只看得到最近的對話紀錄 ----------

LOG_COLUMNS = ["時間", "姓名", "客人說的話", "判斷意圖", "系統回覆", "耗時(ms)"]

def admin_view_logs(token: str):
    # 用 bytes 比較：compare_digest 對含中文的字串會報錯；它也能避免用回應時間猜密碼
    if not ADMIN_TOKEN or not secrets.compare_digest((token or "").encode(), ADMIN_TOKEN.encode()):
        return [["（密碼錯誤，或尚未設定管理密碼）", "", "", "", "", ""]]
    rows = query_all(
        """SELECT created_at, customer_name, user_message, intent, reply, latency_ms
           FROM conversations ORDER BY id DESC LIMIT 50"""
    )
    return [list(r) for r in rows] or [["（目前沒有紀錄）", "", "", "", "", ""]]

admin_demo = gr.Interface(
    fn=admin_view_logs,
    inputs=gr.Textbox(label="管理密碼", type="password"),
    outputs=gr.Dataframe(headers=LOG_COLUMNS, label="最近 50 筆對話"),
    title="管理後台",
    description="輸入管理密碼後，可查看最近的對話紀錄。"
)

app_ui = gr.TabbedInterface([demo, admin_demo], ["客服", "管理後台"])
app_ui.launch()