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

客服 Chatbot POC（情境：髮廊預約與諮詢）

以「Router ＋ 專責 Agent」架構實作，使用 Gemini API、LangGraph、MCP

- 線上 Demo：https://huggingface.co/spaces/xuen0901/booking-chatbot-poc
- 原始碼：https://github.com/jimmy40901/booking-chatbot-poc

## 專案目標

驗證能否以「Router ＋ 專責 Agent」的架構，在單一對話入口同時處理兩類需求：一問一答的諮詢（營業時間、價格、服務說明），以及多步驟的預約流程；並能依客戶身分（新客／熟客／一般舊客）動態調整回覆語氣與優惠提示，作為客服 Chatbot 的 POC。

## 功能

- **Router ＋ 專責 Agent**：Router 判斷使用者意圖，分派給 FAQ Agent 或 Booking Agent
- **FAQ Agent**：以 Tool Calling 查詢營業時間與價格，並用 RAG（Embedding ＋ 餘弦相似度）檢索 50 筆 FAQ；工具由獨立的 MCP Server 提供，一句話內的多個工具呼叫共用同一次連線
- **Booking Agent**：以 LangGraph 實作的引導式預約流程，能從一句話提取服務與日期（今天、明天、後天、禮拜X、下禮拜X），資訊不足時追問，預約到一半插入其他問題，已填的資訊不會遺失
- **預約規則**：以日期為單位；每天上限 5 位客人，同一位客人同一天的多項服務視為同一次到店；同一天同一服務不可重複預約；公休或額滿時說明原因，並建議最多 3 個可預約日期；預約回覆由程式模板產生，不由 LLM 生成
- **資料層**：營業時間、價格、客人、預約與對話紀錄存於 SQLite；營業時間與價格只有這一個來源，FAQ 內容不重複放價格或營業時間
- **客戶身分**：依新客、熟客、一般舊客動態組出 System Prompt，調整語氣與優惠提示
- **管理後台**：檢視對話紀錄；以 `ADMIN_TOKEN` 保護，未設定則完全不開放
- **介面**：Gradio 網頁（線上 Demo），以及 LINE 官方帳號（目前僅限本機執行，見下方「LINE 串接」）
- **檢索評估**：30 題評估題，Top-1 命中率 21/24，Top-3 命中率 24/24（評估題為 AI 草擬，詳見下方「檢索評估」）

## 展示

**工具呼叫（經由 MCP 查詢價格）**

![查詢剪髮價格](assets/01-price-mcp.png)

**一句話觸發多個工具呼叫**

![同時查詢染髮與燙髮價格](assets/02-two-tools.png)

**查不到時承認，不編造答案**

![詢問 FAQ 沒有的內容](assets/03-no-guess.png)

以上為 Gradio 網頁介面；回覆內容由 LLM 生成，每次可能略有不同。

## 系統架構

```
使用者
  ├─ Gradio 網頁（客服頁、管理後台）
  └─ LINE 官方帳號（Webhook 驗簽；目前僅限本機）
        │
        ▼
┌──────────────┐
│    Router    │  Gemini 判斷意圖：faq 或 booking
└──────────────┘
        │
        ├── faq ──────▶ FAQ Agent ──▶ ask()
        │                              ├─ 動態 System Prompt（依客戶身分）
        │                              └─ Gemini Tool Calling
        │                                    │
        │                            MCP Client（stdio）
        │                                    ▼
        │                            MCP Server（獨立行程）
        │                              ├─ check_opening_hours ─▶ SQLite
        │                              ├─ check_service_price ─▶ SQLite
        │                              └─ search_faq ─▶ Embedding ＋ 餘弦相似度（50 筆 FAQ）
        │
        └── booking ──▶ Booking Agent ──▶ booking_app（LangGraph 子圖）
                                           ├─ 提取服務與日期（Gemini；結果由 Python 驗證，日期由 Python 換算）
                                           ├─ 資訊不足：追問服務或日期
                                           ├─ 查空位（SQLite：公休、額滿、重複預約）
                                           ├─ 可預約但沒有姓名：詢問姓名
                                           ├─ 可預約：寫入預約（單一 SQL 原子寫入）
                                           └─ 不可預約：說明原因並建議改期
                                              （預約回覆為程式模板，非 LLM 生成）

所有對話皆寫入 SQLite（對話紀錄），供管理後台檢視
```

系統分為四層：資料層（SQLite、FAQ 與向量檔）、工具層（MCP Server）、能力層（`ask`、`booking_app`）、調度層（Router 與兩個 Agent）；介面（Gradio、LINE）在最上面，與調度層對接。下層不依賴上層，因此新增 Agent 時只需修改調度層，新增介面時不需改動既有流程（LINE 即以此方式接入）。這是邏輯上的分層，程式碼目前主要集中在 `app.py`。

## 架構設計決策

### 為什麼使用 Tool Calling

初期測試發現，若僅依賴 System Prompt 要求 AI 回答特定資訊（如營業時間），AI 在缺乏真實資料來源時會產生看似合理但錯誤的答案（Hallucination）。因此改採 Tool Calling 架構：將營業時間、服務價格等查詢邏輯實作為獨立函式，由 AI 依據使用者意圖判斷並呼叫對應工具，取得真實資料後再生成回覆，避免模型憑空臆測。

### 為什麼加入 RAG

Tool Calling 僅適用於結構化、可明確定義參數的查詢（如查詢特定星期的營業時間）。當客人詢問染後保養、預約規則等描述性問題時，無法事先窮舉所有可能情境並逐一寫成工具。因此導入 RAG：將服務說明文件以 Embedding 轉換為向量，透過餘弦相似度檢索出與問題語意最相關的段落，再交由 LLM 根據檢索結果生成回覆。

### 為什麼把 RAG 包成工具

為了讓系統能在單一對話中同時處理結構化查詢與非結構化問題，將 RAG 檢索流程（`retrieve`）包裝為一個對外僅暴露 `question` 參數的工具（`search_faq`），與其他工具並列於同一份 Tool 清單中。AI 可依據使用者意圖，在 Tool Calling 與 RAG 之間自主判斷呼叫對象，不需額外撰寫規則式的路由邏輯。

### 為什麼用 Router 分派給不同 Agent

系統的兩類需求在互動形式上不同：問答是一問一答，單次處理即可；預約則是多步驟流程，需要記住已取得的資訊（服務項目、日期、是否有空位），並依中間結果分岔。若全部交給單一 `ask` 處理，Prompt 與工具會越來越雜，流程控制也缺乏明確的狀態管理。因此以 Router 先判斷意圖：問答交給 Tool Calling ＋ RAG 的 FAQ Agent，預約交給以 LangGraph State 管理流程的 Booking Agent。各 Agent 專注單一職責，新增 Agent 時只需修改調度層。

### 為什麼用「預約草稿」這種小字典來記憶，而不是把整段對話歷史都丟給 LLM？

預約流程只需要兩項資訊（服務項目、日期），因此以一個小字典（預約草稿）記錄已取得的欄位，每輪只把「當前這句話＋草稿內容」交給 LLM，而不附上完整對話歷史。考量如下：

1. **輸入長度固定**：每次呼叫的輸入量不會隨對話變長而增加，預期成本與延遲較穩定（呼叫次數不變，差別在每次呼叫的輸入量；尚未實測）。
2. **行為可預期**：由程式決定何時補齊、何時清空。預約完成後草稿隨即清空，下一筆預約不會誤用上一筆的資料；若交給 LLM 從歷史中自行判斷，較難保證這點。
3. **容易除錯**：草稿是明確的資料，可直接顯示在介面上（「系統目前記住的預約資訊」），出錯時能立刻看出系統記住了什麼。

代價是記憶範圍僅限這兩個欄位，無法承接一般對話脈絡（例如 FAQ 的追問）。

### 為什麼把工具搬到 MCP Server？

**動機**：希望工具呼叫有一個固定的格式規範，同時讓主程式（`app.py`）專注在系統設計的邏輯上。

**做法**：把營業時間查詢、價格查詢、FAQ 檢索三個工具移到獨立的 `mcp_server.py`，`app.py` 透過 MCP Client（stdio）呼叫。工具的名稱與參數格式由函式簽名與 docstring 自動產生，不必另外手寫一份給工具端。搬移後，移除了 `app.py` 裡重複的工具邏輯，避免同一份邏輯維護兩處。

**預約流程不經過 MCP**：預約流程查營業日時，是由程式直接呼叫函式。預約是固定流程，沒有 LLM 決定要不要查，繞 MCP 只會多花時間。

**失敗時的處理**：MCP 啟動或呼叫失敗，系統回覆「暫時無法查詢」，不退回呼叫本機函式，以免同一份邏輯維護兩處。

**代價**：每次工具呼叫都要啟動一次 Server，本機單次實測約 2.7 到 4.5 秒；同一句話內的多個工具呼叫共用同一次連線。另外還要處理同步程式呼叫非同步 Client 的轉接，以及套件版本相容（所用的 gradio 版本要求 `mcp` 低於 2.0）。

**尚未做到**：給 Gemini 的工具宣告仍是手寫，尚未改成從 Server 自動取得；「工具可被其他程式共用」是架構上的可能，尚未實際用第二個程式驗證；HF 上的延遲尚未量測。功能在 HF 上已驗證可用（價格查詢、FAQ 檢索、單句多工具呼叫）。

## 技術選擇

- LLM：Gemini API（gemini-3.5-flash-lite，免費額度）
- Embedding：gemini-embedding-001
- 流程編排：LangGraph
- 向量運算：NumPy（自行實作餘弦相似度，尚未使用向量資料庫）
- 介面與部署：Gradio、Hugging Face Spaces
- 規劃以 Claude API 做對照比較
- 向量索引：FAQ 向量預先計算後存成 `.npy` 檔（`build_index.py`），啟動時直接載入，不需每次呼叫 Embedding API

## 目前限制與下一步

### 目前限制


- 營業時間、價格、預約與對話紀錄存於 SQLite；資料為模擬資料，FAQ（50 筆）為自行撰寫，評估題為 AI 草擬
- 客戶身分機制已實作於 `ask()`，但多 Agent 版本的介面使用固定身分（一般舊客），尚未開放選擇
- RAG 檢索到正確的 FAQ 後，LLM 生成回答時仍可能補充 FAQ 沒有的內容（例如自行加上未經確認的建議），目前以 System Prompt 約束，尚未系統性評估
- 檢索永遠回傳相似度最高的一筆，沒有門檻；問題在 FAQ 範圍外時仍會取回最接近的一筆
- 預約以日期為單位（不含時段）；「禮拜X」解讀為最近的下一個該星期，與今天同一星期時視為下週；「下禮拜X」為下一週（週一起算）的該天；不處理「大後天」、「下個月」等說法
- 每天上限 5 位客人；同一位客人同一天可預約多項服務，視為同一次到店，只佔一個名額；同一客人同一天同一服務不可重複預約
- 預約不檢查今天是否已過營業時間
- 建議的替代日期只往後找，最多看兩週
- 客戶以姓名查詢，不具備身分驗證，也不處理同名；規劃串接 LINE 後改以 LINE 使用者 ID 辨識
- Hugging Face Space 的檔案系統為暫存性質，重新啟動後預約與對話紀錄會消失
- 預約記憶存在每位使用者各自的 session 中，重新整理頁面即重置，尚未持久化
- 記憶僅限預約流程的兩個欄位（服務、日期），FAQ 問答不保留對話脈絡，例如追問「那燙髮呢？」無法承接上一題
- 工具以 MCP Server 提供，主程式處理工具呼叫時會啟動一次 Server（同一句話的多個工具呼叫共用同一次連線）；本機單次實測約 2.7～4.5 秒，HF 上的功能（價格查詢、FAQ 檢索、單句多工具呼叫）已驗證可用，延遲數字尚未在 HF 上量測
- MCP Server 啟動或呼叫失敗時，系統回覆「暫時無法查詢」，不會退回直接呼叫本機函式，以避免同一份邏輯維護兩處
- 本機（numpy 1.26.4、pandas 2.2.3）與 Hugging Face（numpy 2.5.3、pandas 3.0.6）的套件大版本不同；HF 上已驗證客服問答與工具呼叫可用，評估程式（`eval_score.py`）僅在本機執行過

#### 已知限制的成因與改進方向

**1. 一句話只能提取一項服務**
- 現象：客人說「想預約剪髮跟染髮」，系統只會處理其中一項。
- 原因：預約資訊提取的輸出格式只有單一 `service` 欄位，LLM 無法回傳多項。
- 改進方向：將 `service` 改為清單，並讓後續流程逐項檢查。

**2. Router 一次只選一條路**
- 現象：同一句話同時包含詢問與預約（例如「剪髮多少錢？幫我約明天」），只會走其中一條流程。
- 原因：Router 輸出單一意圖標籤。
- 改進方向：允許輸出多個意圖，依序交給對應 Agent。
- MCP 工具每次呼叫都會重新啟動 Server，約增加 2～3 秒延遲


### 檢索評估

**方法**：以 30 題評估題測試 RAG 檢索，其中 24 題有標準答案、6 題為範圍外問題。每題轉成向量後，與 50 筆 FAQ 以餘弦相似度比較，取前 3 名。評估題由 AI 草擬，非真實客人提問，FAQ 與資料皆為模擬，結果僅供參考。

| 指標 | 結果 |
|---|---|
| Top-1 命中率 | 21 / 24 |
| Top-3 命中率 | 24 / 24 |

**發現**

1. Top-1 與 Top-3 差 3 題：這 3 題中，相似度最高的不是正確 FAQ，但正確 FAQ 都在前 3 名。可能的改進方向（尚未驗證）：把前 3 筆一併交給 LLM 判斷。
2. 範圍外題目的第一名分數為 0.61～0.779，都低於有答案題目的最低分 0.825；但分數最高的離題題目，都與髮廊相關（販售商品、帶寵物入店）。
3. 目前未加相似度門檻：兩組分數的間隔只有約 0.046，樣本僅 6 題範圍外，且分數可能受問法長短影響（推測，未驗證）。另外手動測試 2 題分數最高的離題問題，回覆都承認查不到或指出 FAQ 未提及，但只有 2 題，不構成保證。

**重現方式**：先執行 `eval_retrieval.py`（會呼叫 Gemini embedding 30 次，產生 `eval/eval_vectors.npy`），再執行 `eval_score.py`（只讀本機檔案，不需網路）。


## 快速開始（本機）

**需求**：Python 3.11 或 3.12、[Git LFS](https://git-lfs.com)（FAQ 向量檔以 LFS 儲存）、Gemini API 金鑰。

```
git lfs install
git clone https://github.com/jimmy40901/booking-chatbot-poc.git
cd booking-chatbot-poc
pip install -r requirements.txt gradio==6.29.1
copy .env.example .env
```

- 請用 `git clone`，不要用 Download ZIP；沒有安裝 Git LFS 時，`data/faq_vectors_q.npy` 只會是指標檔，檢索會失敗。
- `gradio==6.29.1` 與 Hugging Face Space 設定的版本相同；此版本要求 `mcp` 低於 2.0，所以 `requirements.txt` 另外鎖定 `mcp==1.30.0`。
- 用文字編輯器打開 `.env`，填入 `GEMINI_API_KEY`（必填）。`ADMIN_TOKEN` 選填，沒設定時管理後台不開放。LINE 的兩個變數只有串接 LINE 時才需要。**金鑰只放 `.env`，不要提交到 Git。**

**啟動**

```
python app.py
```
首次啟動時會自動建立 `booking.db` 並填入模擬資料（營業時間、價格、示範客戶）。
開啟 http://127.0.0.1:7860 。工具由 `mcp_server.py` 提供，會在需要時由程式自動啟動，不必另外執行。

**LINE（選用，目前僅限本機）**：另外需要 LINE 官方帳號、Messaging API，以及 ngrok 之類的公開通道。以 `python -m uvicorn line_webhook:app --port 8000` 啟動，並將 Webhook 網址設為通道網址加上 `/callback`，且開啟 Use webhook。

### 部署紀錄：依賴版本不一致

**現象**：加入 MCP 後推送到 Hugging Face，建置成功但啟動失敗，錯誤為 `ImportError: cannot import name 'genai' from 'google'`。更早一次的 Runtime error 當時沒有錯誤內容，無法確認是否為同一個原因。

**原因**：`requirements.txt` 中 `google-genai` 未鎖定版本，HF 每次重建都會安裝當下最新版（先前的建置紀錄顯示為 2.33.0），本機測試的則是 2.25.0。將 `google-genai` 鎖定為 `2.25.0` 後問題消失，由此確認版本差異是原因；但最新版為何會失敗，尚未查明。

**做法**：鎖定 `google-genai==2.25.0`。其他套件目前仍未鎖版本，同類問題可能再發生，列為待處理。


### LINE 串接

**架構**：LINE 平台以 Webhook 將訊息送到 `line_webhook.py`（FastAPI），驗證簽章後，交給與網頁版相同的 Router＋Agent 流程處理，再用 Reply API 回覆。目前僅在本機執行，以 ngrok 通道對外。需要的環境變數：`LINE_CHANNEL_SECRET`、`LINE_CHANNEL_ACCESS_TOKEN`（只放 `.env`）。

**設計決策：預約時如何取得姓名**

- **現象**：原本的預約流程是為 Gradio 設計的，姓名來自網頁上的獨立欄位。移到 LINE 後沒有這個欄位，流程停在「請填姓名」，客人在聊天室輸入姓名，回覆卻變成一般打招呼（推測被當成新的問答，尚未查證 Router 的判斷），預約無法完成。
- **做法**：不改動 LangGraph 的流程，改在 LINE 這一層處理,好處是網頁版完全不受影響。第一次預約時，在聊天室詢問姓名，並把「LINE 使用者 ID → 姓名」存入 SQLite，之後同一個 LINE 帳號不再詢問。
- **如何判斷正在等待姓名**：預約草稿的服務與日期都已有值，且資料庫尚無此 LINE ID 的姓名。此時下一句話直接視為姓名，不經過 Router。

**已知限制**

- 目前只在本機執行（ngrok），尚未部署
- 預約草稿存在程式記憶體，重啟後消失
- 等待姓名時，30 字內的任何文字都會被當成姓名（例如客人此時改說「改成後天」）
- 姓名無驗證，同名視為同一人
- 只處理文字訊息，貼圖與加好友等事件忽略
- 使用 replyToken 回覆，官方未公布有效時間；本機實測最長約 10 秒內可正常回覆
- 「LINE ID → 姓名」屬個資，目前僅用測試資料，LINE 端尚無隱私說明
- 訊息額度與費用尚未查證


### 下一步規劃

- 以政府資料開放平臺的 FAQ 資料驗證 RAG 檢索品質（建立評估集、比較命中率）
- 將檢索與查詢工具改以 MCP Server 提供
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

修改 `data/faq.csv` 後，需先刪除 `data/faq_vectors_q.npy`，再執行 `python build_index.py` 重新建立索引，否則向量會與 FAQ 內容對不上。