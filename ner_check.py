# -*- coding: utf-8 -*-
"""
NER Validation Stage: проверяет целостность и качество результатов NER.
Чтение:
  30_ner/<book_id>_ner.json
  30_ner/<book_id>_mentions_index.json
Запись:
  30_ner/<book_id>_ner_report.json
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
from collections import Counter
from loguru import logger

def run_stage(paths: Dict[str, Path], cfg: Dict[str, Any]) -> None:
    """
    Проверяет, что каждый mention_id из ner.json
    есть в индексе mentions_index.json, а также
    нет дубликатов mention_id.
    Сохраняет отчёт ner_report.json.
    """
    # Определяем book_id по имени директории ner_dir
    book_id = paths['book_root'].name
    ner_path  = paths["ner_dir"] / f"{book_id}_ner.json"
    idx_path  = paths["ner_dir"] / f"{book_id}_mentions_index.json"

    # Проверяем входные файлы
    if not ner_path.exists():
        logger.error(f"[ner_check] Не найден файл NER: {ner_path}")
        return
    if not idx_path.exists():
        logger.error(f"[ner_check] Не найден индекс упоминаний: {idx_path}")
        return

    # Загружаем данные
    ner_data       = json.loads(ner_path.read_text(encoding="utf-8"))
    mentions_index = json.loads(idx_path.read_text(encoding="utf-8"))

    entities        = ner_data.get("entities", [])
    total_entities  = len(entities)
    total_mentions  = 0

    missing_in_index: List[Tuple[str, str]] = []
    mention_ids: List[str] = []

    # Собираем статистику по упоминаниям
    for ent in entities:
        mentions = ent.get("mentions", [])
        total_mentions += len(mentions)
        for m in mentions:
            mid = m.get("mention_id")
            mention_ids.append(mid)
            key = f"{m['chapter']}::{m['scene']}::{m['sent_id']}"
            idx_list = mentions_index.get(key, [])
            if mid not in idx_list:
                missing_in_index.append((mid, key))

    # Ищем дубли упоминаний
    duplicates = [mid for mid, cnt in Counter(mention_ids).items() if cnt > 1]

    # Среднее упоминаний на сущность
    per_entity   = [len(ent.get("mentions", [])) for ent in entities]
    avg_mentions = (sum(per_entity) / total_entities) if total_entities else 0.0

    # Логируем
    logger.info(f"[ner_check] Всего сущностей: {total_entities}")
    logger.info(f"[ner_check] Всего упоминаний: {total_mentions}")
    logger.info(f"[ner_check] Среднее упоминаний на сущность: {avg_mentions:.2f}")

    if missing_in_index:
        logger.warning(f"[ner_check] Отсутствуют в index.json: {len(missing_in_index)} упоминаний")
        for mid, key in missing_in_index[:5]:
            logger.debug(f"  Missing: {mid} in {key}")

    if duplicates:
        logger.warning(f"[ner_check] Дубли mention_id: {len(duplicates)} элементов")
        for mid in duplicates[:5]:
            logger.debug(f"  Duplicate ID: {mid}")

    # Собираем отчёт
    report = {
        "total_entities":            total_entities,
        "total_mentions":            total_mentions,
        "avg_mentions_per_entity":   avg_mentions,
        "missing_in_index":          len(missing_in_index),
        "duplicates":                len(duplicates),
        "issues_detected":           bool(missing_in_index or duplicates)
    }

    # Сохраняем ner_report.json
    report_path = paths["ner_dir"] / f"{book_id}_ner_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    logger.info(f"[ner_check] Report saved → {report_path.name}")
