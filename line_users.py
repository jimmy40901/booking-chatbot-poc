import sqlite3

from init_db import DB_PATH


def init_line_users(db_path: str = DB_PATH) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS line_users ("
            "line_user_id TEXT PRIMARY KEY, name TEXT NOT NULL)"
        )
        conn.commit()
    finally:
        conn.close()


def get_name(line_user_id: str, db_path: str = DB_PATH) -> str:
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT name FROM line_users WHERE line_user_id = ?", (line_user_id,)
        ).fetchone()
        return row[0] if row else ""
    finally:
        conn.close()


def set_name(line_user_id: str, name: str, db_path: str = DB_PATH) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "INSERT INTO line_users (line_user_id, name) VALUES (?, ?) "
            "ON CONFLICT(line_user_id) DO UPDATE SET name = excluded.name",
            (line_user_id, name.strip()[:30]),
        )
        conn.commit()
    finally:
        conn.close()


if __name__ == "__main__":
    import os
    import tempfile

    path = os.path.join(tempfile.mkdtemp(), "test.db")
    init_line_users(path)
    print("新使用者：", repr(get_name("U-fake-1", path)))
    set_name("U-fake-1", "  測試員  ", path)
    print("存入後：", repr(get_name("U-fake-1", path)))
    set_name("U-fake-1", "小華", path)
    print("改名後：", repr(get_name("U-fake-1", path)))
    print("別人：", repr(get_name("U-fake-2", path)))