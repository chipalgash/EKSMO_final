# io_utils.py
# -*- coding: utf-8 -*-
from pathlib import Path
import json
from typing import Any, Dict

import config  # предполагается, что config.py лежит рядом и экспортирует DATA_DIR и SUBDIRS

DATA_DIR: Path = Path(config.DATA_DIR)
SUBDIRS: Dict[str, str] = config.SUBDIRS

def get_book_dir(book_id: str) -> Path:
    """
    Возвращает корневую папку книги вида DATA_DIR / book_id
    """
    return DATA_DIR / book_id

def ensure_dirs(book_id: str, *keys: str) -> Dict[str, Path]:
    """
    Убеждается, что для книги book_id созданы поддиректории с именами SUBDIRS[key].
    Вернёт словарь key -> Path.
    Если ключа нет в SUBDIRS — бросит KeyError.
    """
    base = get_book_dir(book_id)
    base.mkdir(parents=True, exist_ok=True)

    dirs: Dict[str, Path] = {}
    for key in keys:
        if key not in SUBDIRS:
            raise KeyError(f"Unknown subdirectory key: {key}")
        subdir = base / SUBDIRS[key]
        subdir.mkdir(parents=True, exist_ok=True)
        dirs[key] = subdir

    return dirs

def save_json(obj: Any, path: Path) -> None:
    """
    Сохраняет Python-объект в файл path в формате JSON.
    Если папки не существуют — создаёт их.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_json(path: Path) -> Any:
    """
    Читает JSON-файл path и возвращает распаршенный объект.
    """
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
