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
    # Получаем идентификатор книги и пути
    book_id = paths['book_root'].name
    coref_path = paths['coref_dir'] / f"{book_id}_coref.json"
    preproc_path = paths['preprocess_dir'] / f"{book_id}_preprocessed.json"
    out_dir = paths['contexts_dir']
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{book_id}_contexts.json"

    # Пропустить, если уже есть и не force
    if out_path.exists() and not cfg.get('force', False):
        logger.info(f"[contexts] Результаты есть, пропускаю: {out_path.name}")
        return

    # Загрузка входных данных
    if not coref_path.exists():
        raise FileNotFoundError(f"[contexts] Нет coref-файла: {coref_path}")
    if not preproc_path.exists():
        raise FileNotFoundError(f"[contexts] Нет препроцессинга: {preproc_path}")
    with coref_path.open('r', encoding='utf-8') as f:
        coref_data = json.load(f)
    with preproc_path.open('r', encoding='utf-8') as f:
        prep_data = json.load(f)

    # Параметры из config
    left_win = cfg.get('left_sentences', 2)
    right_win = cfg.get('right_sentences', 1)
    max_ctx = cfg.get('max_contexts_per_char', 100)
    max_chars = cfg.get('max_chars_per_context', 2500)

    # Строим кэш сцен: (chapter, scene) -> {'text', 'sentences', 'offsets'}
    scene_cache: Dict[Tuple[int,int], Dict[str, Any]] = {}
    for ch in prep_data.get('chapters', []):
        ch_id = ch['id']
        for sc in ch.get('scenes', []):
            sc_id = sc['id']
            # Берем либо готовый текст сцены, либо собираем из предложений
            text = sc.get('text') or " ".join(s['text'] for s in sc.get('sentences', []))
            sents = [s['text'] for s in sc.get('sentences', [])]
            # Вычисляем оффсеты предложений
            offs: List[Tuple[int,int]] = []
            pos = 0
            for sent in sents:
                idx = text.find(sent, pos)
                if idx < 0:
                    idx = pos
                offs.append((idx, idx+len(sent)))
                pos = idx + len(sent)
            scene_cache[(ch_id, sc_id)] = {'text': text, 'sentences': sents, 'offsets': offs}

    # Собираем контексты для каждого персонажа
    contexts: List[Dict[str, Any]] = []
    for ent in coref_data.get('entities', []):
        events: List[Dict[str, Any]] = []
        # Сортируем упоминания
        mentions = sorted(
            ent.get('mentions', []),
            key=lambda m: (m['chapter'], m['scene'], m.get('sent_id', -1), m.get('start', -1))
        )
        for m in mentions:
            ch_id = m['chapter']; sc_id = m['scene']; sid = m.get('sent_id')
            cache = scene_cache.get((ch_id, sc_id))
            if not cache:
                continue
            sents = cache['sentences']; offs = cache['offsets']
            # Если sent_id вне диапазона, ищем по смещению
            if sid is None or sid < 0 or sid >= len(sents):
                start = m.get('start', 0)
                sid = 0
                for i, (st, en) in enumerate(offs):
                    if st <= start < en:
                        sid = i; break
            # Окно предложений
            left = max(0, sid - left_win)
            right = min(len(sents)-1, sid + right_win)
            window = sents[left:right+1]
            # Отрезок полного текста
            span_start = offs[left][0]
            span_end = offs[right][1]
            full_text = cache['text'][span_start:span_end]
            # Ограничим длину фрагмента
            if len(full_text) > max_chars:
                full_text = full_text[:max_chars]
            events.append({
                'mention_id': m['mention_id'],
                'chapter': ch_id,
                'scene': sc_id,
                'sent_id': sid,
                'window': window,
                'text': full_text,
                'mention_text': m.get('text'),
                'type': m.get('type', 'name')
            })
            if len(events) >= max_ctx:
                break
        contexts.append({
            'entity_id': ent['id'],
            'norm': ent.get('norm'),
            'gender': ent.get('gender', 'unknown'),
            'aliases': ent.get('aliases', []),
            'contexts': events
        })

    # Сохраняем
    with out_path.open('w', encoding='utf-8') as f:
        json.dump({'book_id': book_id, 'contexts': contexts}, f, ensure_ascii=False, indent=2)
    logger.info(f"[contexts] Сохранено: {out_path.name}")

