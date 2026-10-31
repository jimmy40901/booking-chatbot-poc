import base64
import hashlib
import hmac
import json
import os
from dotenv import load_dotenv
import httpx
import asyncio
from app import chatbot_interface

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

load_dotenv()
app = FastAPI()


def verify_signature(body: bytes, signature: str, channel_secret: str) -> bool:
    """驗證請求真的來自 LINE：對原始內容算 HMAC-SHA256，再與標頭的簽章比對"""
    digest = hmac.new(channel_secret.encode("utf-8"), body, hashlib.sha256).digest()
    expected = base64.b64encode(digest)
    return hmac.compare_digest(expected, signature.encode("utf-8"))

def reply_text(reply_token: str, text: str) -> None:
    """用 replyToken 回覆一則文字訊息；失敗只印狀態，不印內容與金鑰"""
    token = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
    if not token:
        print("缺少 LINE_CHANNEL_ACCESS_TOKEN")
        return
    try:
        r = httpx.post(
            "https://api.line.me/v2/bot/message/reply",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "replyToken": reply_token,
                "messages": [{"type": "text", "text": text[:1000]}],
            },
            timeout=5,
        )
        if r.status_code != 200:
            print("LINE 回覆失敗，狀態碼：", r.status_code)
    except Exception as e:
        print("LINE 回覆失敗：", type(e).__name__)

drafts = {}  # {LINE 使用者 ID: 預約草稿}，只存在記憶體，重啟會消失


def answer(user_id: str, text: str) -> str:
    """把客人的話交給現有的客服流程，回傳要回覆的文字"""
    try:
        draft = drafts.get(user_id, {})
        reply, _, new_draft = chatbot_interface(text[:500], "", draft)
        drafts[user_id] = new_draft
        return reply
    except Exception as e:
        print("客服流程失敗：", type(e).__name__)
        return "抱歉，系統暫時無法回覆，請稍後再試。"


@app.post("/callback")
async def callback(request: Request):
    body = await request.body()
    signature = request.headers.get("x-line-signature", "")
    secret = os.getenv("LINE_CHANNEL_SECRET", "")
    if not secret or not verify_signature(body, signature, secret):
        return JSONResponse({"error": "invalid signature"}, status_code=400)
    events = json.loads(body).get("events", [])
    for event in events:
        message = event.get("message", {})
        if (
            event.get("type") == "message"
            and message.get("type") == "text"
            and event.get("replyToken")
        ):
            user_id = event.get("source", {}).get("userId", "")
            reply = await asyncio.to_thread(answer, user_id, message.get("text", ""))
            reply_text(event["replyToken"], reply)
    return {"events": len(events)}


if __name__ == "__main__":
    from fastapi.testclient import TestClient

    secret = "fake-secret-for-testing"
    os.environ["LINE_CHANNEL_SECRET"] = secret
    client = TestClient(app)

    body = b'{"events":[{"type":"message"}]}'
    sig = base64.b64encode(
        hmac.new(secret.encode("utf-8"), body, hashlib.sha256).digest()
    ).decode("utf-8")

    r = client.post("/callback", content=body, headers={"x-line-signature": sig})
    print("正確簽章：", r.status_code, r.json())
    r = client.post("/callback", content=body + b" ", headers={"x-line-signature": sig})
    print("內容被改：", r.status_code)
    r = client.post("/callback", content=body, headers={"x-line-signature": "abc"})
    print("簽章亂填：", r.status_code)
    r = client.post("/callback", content=body)
    print("沒有簽章：", r.status_code)