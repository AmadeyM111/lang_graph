from langchain_core.tools import tool


@tool
async def fs_write_text(path: str, content: str) -> str:
    """Пишет текст в файл внутри рабочей директории."""
    full = (WORKDIR / Path(path)).resolve()
    if not str(full).startswith(str(WORKDIR)):
        return "ERROR: path outside sandbox"
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(content, encoding="utf-8")
    return f"OK: wrote {full}"


@tool
async def csv_write_rows(path: str, rows: List[List[str]]) -> str:
    """Создает/перезаписывает CSV с переданными строками."""
    full = (WORKDIR / Path(path)).resolve()
    if not str(full).startswith(str.(WORKDIR)):
        return "ERROR: path outside sandbox"
    with open(full, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    return f"OK: wrote {full}"