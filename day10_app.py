# 將 dataclass 和 Tool Calling 結合，讓 AI 可以根據新舊客戶動態調整語氣與策略

import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from dataclasses import dataclass

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

# ---------- 客戶資料模板 ----------

@dataclass
class Customer:
    name: str
    is_new: bool
    visit_count: int = 0
    is_birthday_month: bool = False

# ---------- 真實工具函式 ----------

def check_opening_hours(day: str) -> str:
    """查詢指定星期幾的營業時間"""
    schedule = {
        "禮拜一": "公休",
        "禮拜二": "10:00-20:00",
        "禮拜三": "10:00-20:00",
        "禮拜四": "10:00-20:00",
        "禮拜五": "10:00-21:00",
        "禮拜六": "9:00-21:00",
        "禮拜天": "9:00-18:00",
    }
    return schedule.get(day, "查無此資料")

def check_service_price(service: str) -> str:
    """查詢指定服務項目的價格"""
    prices = {
        "剪髮": "500元",
        "洗髮": "300元",
        "染髮": "1500元",
        "燙髮": "2000元",
    }
    return prices.get(service, "查無此服務項目")

# ---------- 工具說明書(給 Gemini 看) ----------

tools = types.Tool(function_declarations=[
    {
        "name": "check_opening_hours",
        "description": "查詢指定星期幾的營業時間",
        "parameters": {
            "type": "object",
            "properties": {
                "day": {"type": "string", "description": "星期幾，例如：禮拜一、禮拜二"}
            },
            "required": ["day"]
        }
    },
    {
        "name": "check_service_price",
        "description": "查詢指定服務項目的價格",
        "parameters": {
            "type": "object",
            "properties": {
                "service": {"type": "string", "description": "服務項目名稱，例如：剪髮、染髮"}
            },
            "required": ["service"]
        }
    }
])

# ---------- 依客戶身分，動態組出 System Prompt ----------

def build_system_prompt(customer: Customer) -> str:
    base = "你是一間髮廊的客服人員，語氣親切，只回答跟營業時間、服務價格有關的問題。"

    if customer.is_new:
        extra = "這位是新客，請特別介紹一下我們的服務，並歡迎他第一次光臨。"
    elif customer.visit_count >= 5:
        extra = f"這位是熟客（已經來訪 {customer.visit_count} 次），可以用更輕鬆熟悉的語氣對話，並主動提及老客戶有 9 折優惠。"
    else:
        extra = "這位是一般舊客，正常親切應對即可。"

    return base + extra

# ---------- 把整套流程包成可重複呼叫的函式 ----------

def ask(question: str, customer: Customer) -> str:
    system_prompt = build_system_prompt(customer)

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=question,
        config=types.GenerateContentConfig(
            tools=[tools],
            system_instruction=system_prompt
        )
    )
    part = response.candidates[0].content.parts[0]

    if part.function_call is None:
        return response.text

    function_call = part.function_call
    if function_call.name == "check_opening_hours":
        tool_result = check_opening_hours(**function_call.args)
    elif function_call.name == "check_service_price":
        tool_result = check_service_price(**function_call.args)

    final_response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=[
            question,
            response.candidates[0].content,
            types.Content(
                role="user",
                parts=[types.Part.from_function_response(
                    name=function_call.name,
                    response={"result": tool_result}
                )]
            )
        ],
        config=types.GenerateContentConfig(
            tools=[tools],
            system_instruction=system_prompt
        )
    )
    return final_response.text

# ---------- 建立 Chatbot 介面 ----------

def chatbot_interface(question: str, customer_type: str, visit_count: int) -> str:
    if customer_type == "新客":
        customer = Customer(name="訪客", is_new=True)
    else:
        customer = Customer(name="訪客", is_new=False, visit_count=visit_count)

    return ask(question, customer)

import gradio as gr

demo = gr.Interface(
    fn=chatbot_interface,
    inputs=[
        gr.Textbox(label="請輸入您的問題"),
        gr.Radio(["新客", "舊客"], label="客戶身分"),
        gr.Number(label="來訪次數（舊客才需要填）", value=0)
    ],
    outputs=gr.Textbox(label="客服回覆"),
    title="髮廊客服 Chatbot POC",
    description="輸入問題，選擇客戶身分，體驗 AI 根據身分動態調整回應策略"
)

demo.launch()