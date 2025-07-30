# -*- coding: utf-8 -*-
"""
Стадия Contexts: сбор контекстов упоминаний персонажей.
Чтение:
  50_coref/<book_id>_coref.json
  20_preprocessed/<book_id>_preprocessed.json
Запись:
  70_contexts/<book_id>_contexts.json
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
from loguru import logger

def run_stage(paths: Dict[str, Path], cfg: Dict[str, Any]) -> None:
    book_id      = paths["book_root"].name
    coref_path   = paths["coref_dir"]     / f"{book_id}_coref.json"
    preproc_path = paths["preprocess_dir"] / f"{book_id}_preprocessed.json"
    out_dir      = paths["contexts_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path     = out_dir / f"{book_id}_contexts.json"

    # Пропустить, если уже есть и не force
    if out_path.exists() and not cfg.get("force", False):
        logger.info(f"[contexts] Есть, пропускаю: {out_path.name}")
        return

    # Загрузка входных данных
    if not coref_path.exists():
        raise FileNotFoundError(f"[contexts] Нет coref‑файла: {coref_path}")
    if not preproc_path.exists():
        raise FileNotFoundError(f"[contexts] Нет препроцессинга: {preproc_path}")
    coref_data = json.loads(coref_path.read_text(encoding="utf-8"))
    prep_data  = json.loads(preproc_path.read_text(encoding="utf-8"))

    # Параметры из config
    left_win  = cfg.get("left_sentences", 2)
    right_win = cfg.get("right_sentences", 1)
    max_ctx   = cfg.get("max_contexts_per_char", 100)
    max_chars = cfg.get("max_chars_per_context", 2500)

    # Строим кэш сцен: (chapter, scene) -> {text, sentences, offsets}
    scene_cache: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for ch in prep_data.get("chapters", []):
        ch_id = ch["id"]
        for sc in ch.get("scenes", []):
            sc_id = sc["id"]
            text  = sc.get("text") or " ".join(s["text"] for s in sc.get("sentences", []))
            sents = [s["text"] for s in sc.get("sentences", [])]
            offs: List[Tuple[int, int]] = []
            pos = 0
            for sent in sents:
                idx = text.find(sent, pos)
                if idx < 0:
                    idx = pos
                offs.append((idx, idx + len(sent)))
                pos = idx + len(sent)
            scene_cache[(ch_id, sc_id)] = {
                "text":      text,
                "sentences": sents,
                "offsets":   offs
            }

    # Собираем контексты для каждого персонажа
    results: List[Dict[str, Any]] = []
    for ent in coref_data.get("entities", []):
        events: List[Dict[str, Any]] = []
        mentions = sorted(
            ent.get("mentions", []),
            key=lambda m: (m["chapter"], m["scene"], m.get("sent_id", -1), m.get("start", -1))
        )
        for m in mentions:
            ch_id, sc_id = m["chapter"], m["scene"]
            cache = scene_cache.get((ch_id, sc_id))
            if not cache:
                continue

            sents, offs = cache["sentences"], cache["offsets"]
            sid = m.get("sent_id")
            if sid is None or not (0 <= sid < len(sents)):
                # Ищем предложение по смещению
                start = m.get("start", 0)
                for i, (st, en) in enumerate(offs):
                    if st <= start < en:
                        sid = i
                        break
                else:
                    sid = 0

            left  = max(0, sid - left_win)
            right = min(len(sents) - 1, sid + right_win)
            window = sents[left : right + 1]

            span_start = offs[left][0]
            span_end   = offs[right][1]
            fragment   = cache["text"][span_start : span_end]
            if len(fragment) > max_chars:
                fragment = fragment[:max_chars]

            events.append({
                "mention_id":   m.get("mention_id"),
                "chapter":      ch_id,
                "scene":        sc_id,
                "sent_id":      sid,
                "window":       window,
                "text":         fragment,
                "mention_text": m.get("text"),
                "type":         m.get("type", "name")
            })
            if len(events) >= max_ctx:
                break

        results.append({
            "entity_id": ent.get("id"),
            "norm":      ent.get("norm"),
            "gender":    ent.get("gender", "unknown"),
            "aliases":   ent.get("aliases", []),
            "events":    events
        })

    # Сохраняем результаты
    out_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    logger.info(f"[contexts] Сохранено: {out_path.name}")
