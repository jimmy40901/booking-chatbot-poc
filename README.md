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

## 專案目標

驗證 LLM 能否根據客戶身分（新客／熟客／一般舊客）動態調整對話策略與回覆內容，包含語氣轉換與優惠資訊提示，作為客服 Chatbot POC 的核心能力驗證。

## 功能

- 透過 Tool Calling 查詢營業時間、服務價格
- 透過 RAG（Embedding + 餘弦相似度）檢索服務說明文件，回答非結構化問題（如染後保養、預約規則）
- 根據客戶身分（新客／舊客／VIP）動態調整 System Prompt，產生不同語氣與內容的回覆
- Gradio 網頁介面，可即時互動測試

## 架構設計

初期測試發現，若僅依賴 System Prompt 要求 AI 回答特定資訊（如營業時間），AI 在缺乏真實資料來源時會產生看似合理但錯誤的答案（Hallucination）。因此改採 Tool Calling 架構：將營業時間、服務價格等查詢邏輯實作為獨立函式，由 AI 依據使用者意圖判斷並呼叫對應工具，取得真實資料後再生成回覆，避免模型憑空臆測。

然而，Tool Calling 僅適用於結構化、可明確定義參數的查詢（如查詢特定星期的營業時間）。當客人詢問染後保養、預約規則等描述性問題時，無法事先窮舉所有可能情境並逐一寫成工具。因此導入 RAG：將服務說明文件以 Embedding 轉換為向量，透過餘弦相似度檢索出與問題語意最相關的段落，再交由 LLM 根據檢索結果生成回覆。

為了讓系統能在單一對話中，同時處理結構化查詢與非結構化問題，進一步將 RAG 檢索流程（`retrieve`）包裝為一個對外僅暴露 `question` 參數的工具（`search_faq`），與其他工具並列於同一份 Tool 清單中。如此一來，AI 可依據使用者意圖，在 Tool Calling 與 RAG 之間自主判斷呼叫對象，不需額外撰寫規則式的路由邏輯。

## 技術選擇

- LLM：Gemini API（gemini-3.5-flash-lite，免費額度）
- 介面：Gradio
- （未來規劃接 Claude API 做比較，因為目標公司主要使用 Claude Code）

## 目前限制與下一步

目前資料是 mock 資料，客戶身分由使用者手動選擇，不是真的查資料庫。接下來規劃：

- Phase 2：加入 RAG，回答更複雜的 FAQ
- Phase 3：多 Agent 架構比較
- Phase 5：接真實資料庫、LINE API

## 如何執行

\`\`\`
pip install -r requirements.txt
python app.py
\`\`\`
