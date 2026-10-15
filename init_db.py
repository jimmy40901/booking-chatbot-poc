import os
import sqlite3

# 資料庫檔案放在這支程式旁邊，不管從哪裡執行都找得到
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "booking.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS services (
    name  TEXT PRIMARY KEY,
    price INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS business_hours (
    weekday TEXT PRIMARY KEY,
    hours   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS customers (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    visit_count INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS appointments (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id INTEGER NOT NULL REFERENCES customers(id),
    service     TEXT NOT NULL REFERENCES services(name),
    date        TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'confirmed'
);
"""

# 以下都是模擬資料(原本寫死在 app.py 的字典，現在搬進資料庫)
SERVICES = [("剪髮", 500), ("洗髮", 300), ("染髮", 1500), ("燙髮", 2000)]

BUSINESS_HOURS = [
    ("禮拜一", "公休"),
    ("禮拜二", "10:00-20:00"),
    ("禮拜三", "10:00-20:00"),
    ("禮拜四", "10:00-20:00"),
    ("禮拜五", "10:00-21:00"),
    ("禮拜六", "9:00-21:00"),
    ("禮拜天", "9:00-18:00"),
]

CUSTOMERS = [("小明", 0), ("小華", 6), ("小美", 2)]


def init_db(reset: bool = False) -> None:
    """建立資料表，並在資料表是空的時候塞入初始資料。reset=True 會先刪掉舊的資料庫。"""
    if reset and os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA)

    # 只在還沒有資料時才塞，重複執行不會產生重複的資料
    if conn.execute("SELECT COUNT(*) FROM services").fetchone()[0] == 0:
        conn.executemany("INSERT INTO services (name, price) VALUES (?, ?)", SERVICES)
        conn.executemany("INSERT INTO business_hours (weekday, hours) VALUES (?, ?)", BUSINESS_HOURS)
        conn.executemany("INSERT INTO customers (name, visit_count) VALUES (?, ?)", CUSTOMERS)

    conn.commit()
    conn.close()


if __name__ == "__main__":
    init_db(reset=True)

    # 建好之後，實際查一次看看
    conn = sqlite3.connect(DB_PATH)
    for table in ["services", "business_hours", "customers", "appointments"]:
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"{table}: {count} 筆")

    price = conn.execute("SELECT price FROM services WHERE name = ?", ("染髮",)).fetchone()
    print("染髮價格：", price)

    missing = conn.execute("SELECT price FROM services WHERE name = ?", ("美甲",)).fetchone()
    print("查不到的服務：", missing)
    conn.close()
