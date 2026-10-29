import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

eval_df = pd.read_csv("eval/eval_set.csv", encoding="utf-8-sig")

vectors = []
for i, question in enumerate(eval_df["question"]):
    result = client.models.embed_content(model="gemini-embedding-001", contents=question)
    vectors.append(result.embeddings[0].values)
    print("完成", i + 1, "/", len(eval_df))

np.save("eval/eval_vectors.npy", np.array(vectors))
print("已存檔")