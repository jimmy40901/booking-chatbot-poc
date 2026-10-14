import os
import time
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

faq = pd.read_csv("data/faq.csv", encoding="utf-8-sig")

def embed_one(text, retries=5):
    for attempt in range(retries):
        try:
            result = client.models.embed_content(
                model="gemini-embedding-001",
                contents=text
            )
            return result.embeddings[0].values
        except Exception as e:
            wait = 5 * (attempt + 1)
            print(f"失敗：{str(e)[:80]}，{wait} 秒後重試...")
            time.sleep(wait)
    raise RuntimeError("多次重試仍失敗")

def build(name, texts):
    path = f"data/faq_vectors_{name}.npy"
    if os.path.exists(path):
        print(f"{path} 已存在，跳過")
        return
    vectors = []
    for text in texts:
        vectors.append(embed_one(text))
        time.sleep(0.7)
    np.save(path, np.array(vectors))
    print(f"{path} 完成")

build("q", faq["question"].tolist())
#build("qa", (faq["question"] + "：" + faq["answer"]).tolist())