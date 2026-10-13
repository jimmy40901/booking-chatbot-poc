---
title: Booking Chatbot Poc
emoji: 🌖
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: 6.29.1
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
---

# Booking Chatbot POC

蜂巢式多 Agent 的髮廊客服 Chatbot（Gemini API ＋ LangGraph）

- 線上 Demo：https://huggingface.co/spaces/xuen0901/booking-chatbot-poc
- 原始碼：https://github.com/jimmy40901/booking-chatbot-poc

## 專案目標

驗證能否以「Router ＋ 專責 Agent」的架構，在單一對話入口同時處理兩類需求：一問一答的諮詢（營業時間、價格、服務說明），以及多步驟的預約流程；並能依客戶身分（新客／熟客／一般舊客）動態調整回覆語氣與優惠提示，作為客服 Chatbot 的 POC。

## 功能

- 蜂巢式多 Agent 架構：Router 自動判斷使用者意圖，分派給對應的 Agent
- FAQ Agent：結合 Tool Calling（查營業時間、服務價格）與 RAG（Embedding ＋ 餘弦相似度檢索服務說明文件）
- Booking Agent：以 LangGraph 實作的引導式預約流程，能從對話中提取服務項目、處理意圖不明確的情況、依空位結果分岔
- 依客戶身分（新客／熟客／一般舊客）動態組出 System Prompt，調整語氣與優惠提示
- Gradio 網頁介面，可即時互動測試

## 系統架構

```
使用者說一句話
      │
      ▼
┌─────────────┐
│   Router    │  判斷意圖（booking 或 faq）
└─────────────┘
      │
      ├── faq ──────▶ FAQ Agent ───▶ ask()
      │                               ├─ Tool Calling（營業時間、價格）
      │                               ├─ RAG（FAQ 文件檢索）
      │                               └─ 動態 System Prompt
      │
      └── booking ──▶ Booking Agent ──▶ booking_app（LangGraph 子圖）
                                        ├─ 提取服務項目
                                        ├─ 查空位
                                        └─ 確認預約 / 建議改期
```

系統分為三層：資料層（客戶模板、工具函式、FAQ 資料）、能力層（`ask`、`booking_app`）、調度層（Router 與兩個 Agent）。下層不依賴上層，因此新增 Agent 時只需修改調度層。

## 架構設計決策

### 為什麼使用 Tool Calling

初期測試發現，若僅依賴 System Prompt 要求 AI 回答特定資訊（如營業時間），AI 在缺乏真實資料來源時會產生看似合理但錯誤的答案（Hallucination）。因此改採 Tool Calling 架構：將營業時間、服務價格等查詢邏輯實作為獨立函式，由 AI 依據使用者意圖判斷並呼叫對應工具，取得真實資料後再生成回覆，避免模型憑空臆測。

### 為什麼加入 RAG

Tool Calling 僅適用於結構化、可明確定義參數的查詢（如查詢特定星期的營業時間）。當客人詢問染後保養、預約規則等描述性問題時，無法事先窮舉所有可能情境並逐一寫成工具。因此導入 RAG：將服務說明文件以 Embedding 轉換為向量，透過餘弦相似度檢索出與問題語意最相關的段落，再交由 LLM 根據檢索結果生成回覆。

### 為什麼把 RAG 包成工具

為了讓系統能在單一對話中同時處理結構化查詢與非結構化問題，將 RAG 檢索流程（`retrieve`）包裝為一個對外僅暴露 `question` 參數的工具（`search_faq`），與其他工具並列於同一份 Tool 清單中。AI 可依據使用者意圖，在 Tool Calling 與 RAG 之間自主判斷呼叫對象，不需額外撰寫規則式的路由邏輯。

### 為什麼用 Router 分派給不同 Agent

系統的兩類需求在互動形式上不同：問答是一問一答，單次處理即可；預約則是多步驟流程，需要記住已取得的資訊（服務項目、日期、是否有空位），並依中間結果分岔。若全部交給單一 `ask` 處理，Prompt 與工具會越來越雜，流程控制也缺乏明確的狀態管理。因此以 Router 先判斷意圖：問答交給 Tool Calling ＋ RAG 的 FAQ Agent，預約交給以 LangGraph State 管理流程的 Booking Agent。各 Agent 專注單一職責，新增 Agent 時只需修改調度層。

## 技術選擇

- LLM：Gemini API（gemini-3.5-flash-lite，免費額度）
- Embedding：gemini-embedding-001
- 流程編排：LangGraph
- 向量運算：NumPy（自行實作餘弦相似度，尚未使用向量資料庫）
- 介面與部署：Gradio、Hugging Face Spaces
- 規劃以 Claude API 做對照比較

## 目前限制與下一步

### 目前限制

- 營業時間、價格、FAQ（5 筆）與空位判斷皆為模擬資料，尚未連接資料庫
- 預約日期固定為禮拜二，尚未讓使用者指定
- 客戶身分機制已實作於 `ask()`，但多 Agent 版本的介面使用固定身分（一般舊客），尚未開放選擇
- 每次提問獨立處理，沒有跨輪對話記憶；意圖不明確時僅提示重新輸入，尚未形成多輪追問

### 下一步規劃

- 以政府資料開放平臺的 FAQ 資料驗證 RAG 檢索品質（建立評估集、比較命中率）
- 將檢索與查詢工具改以 MCP Server 提供
- 以資料庫（模擬資料）取代寫死的字典，並實作預約時段衝突檢查
- 串接 LINE
- 部署至雲端平台

## 如何執行

```bash
pip install -r requirements.txt
```

在專案根目錄建立 `.env` 檔，內容為：

```
GEMINI_API_KEY=你的金鑰
```

接著執行：

```bash
python app.py
```

部署到 Hugging Face Spaces 時，請在 Space 的 Settings 中新增名為 `GEMINI_API_KEY` 的 Secret，不需要上傳 `.env`。