import numpy as np
import pandas as pd

eval_df = pd.read_csv("eval/eval_set.csv", encoding="utf-8-sig")
eval_vectors = np.load("eval/eval_vectors.npy")
faq_vectors = np.load("data/faq_vectors_q.npy")


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


top3_list = []
for question_vector in eval_vectors:
    scores = [cosine_similarity(question_vector, fv) for fv in faq_vectors]
    order = np.argsort(scores)[::-1][:3]
    top3_list.append([(int(i) + 1, round(float(scores[i]), 3)) for i in order])

answerable = eval_df["type"] != "out_of_scope"
top1_hits = 0
top3_hits = 0
misses = []
for k in range(len(eval_df)):
    if not answerable[k]:
        continue
    expected = int(eval_df["expected_faq_id"][k])
    ids = [faq_id for faq_id, score in top3_list[k]]
    if ids[0] == expected:
        top1_hits += 1
    if expected in ids:
        top3_hits += 1
    if ids[0] != expected:
        misses.append((int(eval_df["eval_id"][k]), expected, top3_list[k]))

total = int(answerable.sum())
print("Top-1：", top1_hits, "/", total)
print("Top-3：", top3_hits, "/", total)
for m in misses:
    print("第一名答錯：eval_id", m[0], "正確", m[1], "前三名", m[2])

oos_scores = [top3_list[k][0][1] for k in range(len(eval_df)) if not answerable[k]]
hit_scores = [top3_list[k][0][1] for k in range(len(eval_df)) if answerable[k]]
print("範圍外題目的第一名分數：", sorted(oos_scores, reverse=True))
print("有答案題目的第一名分數範圍：", min(hit_scores), "～", max(hit_scores))

oos_rows = sorted(
    [(top3_list[k][0][1], int(eval_df["eval_id"][k]), eval_df["question"][k])
     for k in range(len(eval_df)) if not answerable[k]],
    reverse=True,
)
print("範圍外分數最高的兩題：", oos_rows[:2])

hit_rows = sorted(
    [(top3_list[k][0][1], int(eval_df["eval_id"][k]), eval_df["type"][k], eval_df["question"][k])
     for k in range(len(eval_df)) if answerable[k]]
)
print("有答案分數最低的兩題：", hit_rows[:2])