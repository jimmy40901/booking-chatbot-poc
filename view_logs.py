"""查看本機 booking.db 裡的對話紀錄。

用法：
    python view_logs.py              顯示統計與最近 20 筆
    python view_logs.py -n 50        顯示最近 50 筆
    python view_logs.py --csv logs_export.csv   匯出全部紀錄成 CSV

注意：匯出的 CSV 含有使用者輸入的內容，請不要 commit 進 Git(已建議加入 .gitignore)。
"""
import argparse
import csv
import sqlite3

from init_db import DB_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description="查看對話紀錄")
    parser.add_argument("-n", type=int, default=20, help="顯示最近幾筆(預設 20)")
    parser.add_argument("--csv", metavar="檔名", help="匯出全部紀錄成 CSV 檔")
    args = parser.parse_args()

    conn = sqlite3.connect(DB_PATH)
    try:
        total = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
        print(f"對話紀錄共 {total} 筆")
        if total == 0:
            return

        print("\n各意圖的筆數：")
        for intent, count in conn.execute(
            "SELECT intent, COUNT(*) FROM conversations GROUP BY intent ORDER BY COUNT(*) DESC"
        ):
            print(f"  {intent or '(未判斷)'}: {count}")

        avg = conn.execute("SELECT AVG(latency_ms) FROM conversations").fetchone()[0]
        print(f"\n平均回應時間：{avg:.0f} ms")

        print(f"\n最近 {args.n} 筆：")
        for created_at, name, message, intent, reply, latency in conn.execute(
            """SELECT created_at, customer_name, user_message, intent, reply, latency_ms
               FROM conversations ORDER BY id DESC LIMIT ?""",
            (args.n,),
        ):
            print(f"[{created_at}] {name or '(未填姓名)'}｜{intent}｜{latency}ms")
            print(f"  問：{message}")
            print(f"  答：{reply[:80]}{'…' if len(reply) > 80 else ''}")

        if args.csv:
            rows = conn.execute(
                """SELECT created_at, session_id, customer_name, user_message, intent, reply, latency_ms
                   FROM conversations ORDER BY id"""
            ).fetchall()
            with open(args.csv, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow(["created_at", "session_id", "customer_name", "user_message", "intent", "reply", "latency_ms"])
                writer.writerows(rows)
            print(f"\n已匯出 {len(rows)} 筆到 {args.csv}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
