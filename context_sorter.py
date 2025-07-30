# -*- coding: utf-8 -*-
"""
Сортировка контекстов перед суммаризацией:
  - по убыванию длины списка events
  - optional: обрезка до top_chars
Чтение/запись: 70_contexts/<book_id>_contexts.json
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List
from loguru import logger

def run_stage(paths: Dict[str, Path], cfg: Dict[str, Any]) -> None:
    book_id   = paths["book_root"].name
    ctx_path  = paths["contexts_dir"]  / f"{book_id}_contexts.json"

    # Проверяем, включён ли этап
    sc_cfg = cfg.get("sort_contexts", {})
    if not sc_cfg.get("enabled", True):
        logger.info(f"[sort_contexts] Отключено в конфигах — пропускаем")
        return

    if not ctx_path.exists():
        logger.error(f"[sort_contexts] Не найден файл контекстов: {ctx_path}")
        return

    # Загружаем
    raw = json.loads(ctx_path.read_text(encoding="utf-8"))
    # unwrap if needed
    if isinstance(raw, dict) and "contexts" in raw:
        ctx_list: List[Dict[str,Any]] = raw["contexts"]
    elif isinstance(raw, list):
        ctx_list = raw
    else:
        logger.error(f"[sort_contexts] Неподдерживаемый формат файла {ctx_path}")
        return

    total_before = len(ctx_list)
    logger.info(f"[sort_contexts] Входных персонажей: {total_before}")

    # Сортируем
    ctx_list.sort(key=lambda ent: len(ent.get("events", [])), reverse=True)

    # Обрезаем, если надо
    top_n = sc_cfg.get("top_chars")
    if isinstance(top_n, int) and top_n > 0:
        ctx_list = ctx_list[:top_n]
        logger.info(f"[sort_contexts] Оставляем топ-{top_n} ⇒ {len(ctx_list)} персонажей")

    # Записываем обратно
    out_data = {"contexts": ctx_list} if isinstance(raw, dict) else ctx_list
    ctx_path.write_text(json.dumps(out_data, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"[sort_contexts] Перезаписали {ctx_path.name}")
