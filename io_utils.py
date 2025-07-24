from pathlib import Path
import json
from .config import DATA_DIR, SUBDIRS

def get_book_dir(book_id: str) -> Path:
    return DATA_DIR / book_id

def ensure_dirs(book_id: str, *keys: str) -> dict[str, Path]:
    base = get_book_dir(book_id)
    base.mkdir(parents=True, exist_ok=True)
    res = {}
    for k in keys:
        sub = base / SUBDIRS[k]
        sub.mkdir(exist_ok=True)
        res[k] = sub
    return res

def save_json(obj, path: Path):
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)