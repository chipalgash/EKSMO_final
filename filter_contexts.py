# -*- coding: utf-8 -*-
"""
Стадия FilterContexts: оставляем только топ‑N персонажей по числу events.
Чтение:
  70_contexts/<book_id>_contexts.json
Запись:
  72_filtered_contexts/<book_id>_contexts.json
"""
import json
from pathlib import Path
from loguru import logger
from typing import Dict, Any

def run_stage(paths: Dict[str, Path], cfg: Dict[str, Any]) -> None:
    book_id  = paths["book_root"].name
    in_path  = paths["contexts_dir"] / f"{book_id}_contexts.json"
    out_dir  = paths["filter_contexts_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / in_path.name

    if not in_path.exists():
        logger.error(f"[filter_contexts] Входной файл не найден: {in_path}")
        return

    data = json.loads(in_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        logger.error(f"[filter_contexts] Ожидался список, получил {type(data)}")
        return

    top_n = cfg.get("top_chars", 5)
    filtered = sorted(
        data,
        key=lambda ent: len(ent.get("events", [])),
        reverse=True
    )[:top_n]

    out_path.write_text(
        json.dumps(filtered, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    logger.info(f"[filter_contexts] {len(data)} → {len(filtered)} сохранено в {out_path.name}")
