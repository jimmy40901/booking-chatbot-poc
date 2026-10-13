import gradio as gr

try:
    import spaces

    @spaces.GPU
    def _warmup():
        return "ok"
except ImportError:
    pass
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from dataclasses import dataclass
from typing import TypedDict
from langgraph.graph import StateGraph, END
import numpy as np

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

def check_opening_hours(day: str) -> str:
    """查詢指定星期幾的營業時間"""
    schedule = {
        "禮拜一": "公休", "禮拜二": "10:00-20:00", "禮拜三": "10:00-20:00",
        "禮拜四": "10:00-20:00", "禮拜五": "10:00-21:00",
        "禮拜六": "9:00-21:00", "禮拜天": "9:00-18:00",
    }
    return schedule.get(day, "查無此資料")

def check_service_price(service: str) -> str:
    """查詢指定服務項目的價格"""
    prices = {"剪髮": "500元", "洗髮": "300元", "染髮": "1500元", "燙髮": "2000元"}
    return prices.get(service, "查無此服務項目")

faq_list = [
    "我們的剪髮服務包含洗髮、剪髮、吹整，約需 1 小時",
    "染髮後建議 48 小時內避免洗頭，讓染劑完全附著",
    "燙髮後一週內請勿使用電棒夾，以免影響捲度",
    "初次來店建議先預約，假日現場候位時間較長",
    "我們使用的染劑為日系進口品牌，低敏配方，但仍建議事先告知過敏史",
]

faq_vectors = []
for faq in faq_list:
    result = client.models.embed_content(model="gemini-embedding-001", contents=faq)
    faq_vectors.append(result.embeddings[0].values)

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def retrieve(question: str, faq_list: list[str], faq_vectors: list) -> str:
    question_result = client.models.embed_content(model="gemini-embedding-001", contents=question)
    question_vector = question_result.embeddings[0].values
    best_score = -1
    best_faq = ""
    for i in range(len(faq_list)):
        score = cosine_similarity(question_vector, faq_vectors[i])
        if score > best_score:
            best_score = score
            best_faq = faq_list[i]
    return best_faq

def search_faq(question: str) -> str:
    """當客人詢問染髮後注意事項、燙髮保養、預約規則、染劑成分等非制式問題時，搜尋相關的服務說明文件"""
    return retrieve(question, faq_list, faq_vectors)

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
    base = "你是一間髮廊的客服人員，語氣親切，只回答跟營業時間、服務價格有關的問題。"
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

class BookingState(TypedDict):
    user_message: str
    service: str
    day: str
    is_available: bool

def extract_service(state: BookingState) -> dict:
    user_input = state['user_message']
    prompt = f"""從以下客人的話中，判斷他想預約的服務項目，只能是：剪髮、洗髮、染髮、燙髮 其中一種。
如果判斷不出來，回傳「不明確」。
只回傳服務項目這兩三個字，不要有其他文字。

客人的話：{user_input}
"""
    response = client.models.generate_content(model="gemini-3.5-flash-lite", contents=prompt)
    service = response.text.strip()
    print(f"判斷出的服務項目：{service}")
    return {"service": service}

def ask_clarification(state: BookingState) -> dict:
    print("不好意思，請問您想預約剪髮、洗髮、染髮還是燙髮呢？")
    return {}

def check_availability(state: BookingState) -> dict:
    print(f"查詢 {state['day']} 是否有空位...")
    available = state['day'] != "禮拜一"
    return {"is_available": available}

def confirm_booking(state: BookingState) -> dict:
    print(f"已為您確認預約：{state['service']}，{state['day']}")
    return {}

def suggest_reschedule(state: BookingState) -> dict:
    print(f"很抱歉，{state['day']} 公休，建議您改約其他時間")
    return {}

def route_after_extract(state: BookingState) -> str:
    return "ask_clarification" if state['service'] == "不明確" else "check_availability"

def route_after_check(state: BookingState) -> str:
    return "confirm_booking" if state['is_available'] else "suggest_reschedule"

booking_graph = StateGraph(BookingState)
booking_graph.add_node("extract_service", extract_service)
booking_graph.add_node("ask_clarification", ask_clarification)
booking_graph.add_node("check_availability", check_availability)
booking_graph.add_node("confirm_booking", confirm_booking)
booking_graph.add_node("suggest_reschedule", suggest_reschedule)
booking_graph.set_entry_point("extract_service")
booking_graph.add_conditional_edges("extract_service", route_after_extract, {
    "ask_clarification": "ask_clarification", "check_availability": "check_availability"
})
booking_graph.add_edge("ask_clarification", END)
booking_graph.add_conditional_edges("check_availability", route_after_check, {
    "confirm_booking": "confirm_booking", "suggest_reschedule": "suggest_reschedule"
})
booking_graph.add_edge("confirm_booking", END)
booking_graph.add_edge("suggest_reschedule", END)

booking_app = booking_graph.compile()

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

def router(state: HiveState) -> dict:
    user_input = state['user_message']
    prompt = f"""判斷客人這句話的意圖，只能回答以下兩種之一：
- booking：如果客人想要預約、約時間
- faq：如果客人是在詢問問題（價格、營業時間、注意事項等）

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

def booking_agent(state: HiveState) -> dict:
    booking_initial_state = {
        "user_message": state['user_message'],
        "service": "", "day": "禮拜二", "is_available": False
    }
    booking_result = booking_app.invoke(booking_initial_state)

    if booking_result['service'] == "不明確":
        response_text = "不好意思，請問您想預約剪髮、洗髮、染髮還是燙髮呢？"
    elif booking_result['is_available']:
        response_text = f"已為您確認預約：{booking_result['service']}，{booking_result['day']}"
    else:
        response_text = f"很抱歉，{booking_result['day']} 公休，建議您改約其他時間"

    return {"final_response": response_text}

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

def chatbot_interface(question: str) -> str:
    result = hive_app.invoke({
        "user_message": question,
        "intent": "",
        "service": "",
        "day": "",
        "is_available": False,
        "final_response": ""
    })
    return result['final_response']

demo = gr.Interface(
    fn=chatbot_interface,
    inputs=gr.Textbox(label="請輸入您的問題（可以問價格/營業時間/保養問題，也可以直接說想預約）"),
    outputs=gr.Textbox(label="客服回覆"),
    title="髮廊客服 Multi-Agent Chatbot",
    description="蜂巢式多 Agent 架構：Router 自動判斷意圖，分派給 FAQ Agent（Tool Calling + RAG）或 Booking Agent（LangGraph 引導式流程）"
)

demo.launch()