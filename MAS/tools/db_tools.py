@tool
async def sqlite_execute(db_path: str, sql: str, params: Optional[List] = None) -> str:
    """Выполняет SQL (DDL/DML). Возвращает 'rows affected'."""
    full = (WORKDIR / Path(db_path)).resolve()
    if not str(full).startswith(str(WORKDIR)):
        return "ERROR: path outside sandbox"
    full.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(full)
    try:
        cur = conn.cursor()
        cur.execute(sql, params or [])
        conn.commit()
        return f"OK: {cur.rowcount} rows affected"
    finally:
        conn.close()

@tool 
async def sqlite_query(db_path: str, sql: str, params: Optional[List] = None) -> List[List[str]]:
    """SELECT-запрос, возвращает строки как списки строк."""
    full = (WORKDIR / Path(db_path)).resolve()
    if not str(full).startswith(str(WORKDIR)) or not full.exists():
        return [["ERROR", "db not found or outside sandbox"]]
    conn = sqlite3.connect(full)
    try:
        cur = conn.cursor()
        cur.execute(sql, params or [])
        rows = cur.fetchall()
        return [[str(x) for x in roew] for row in rows]
    finally:
        conn.close()