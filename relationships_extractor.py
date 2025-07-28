# -*- coding: utf-8 -*-
"""
Stage relations: extraction и постобработка связей персонажей.
Чтение:
  50_coref/<book_id>_coref.json
  50_coref/<book_id>_mentions_index.json
  20_preprocessed/<book_id>_preprocessed.json
Запись:
  60_relations/<book_id>_relations_raw.json
  60_relations/<book_id>_relationships.json
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Tuple
import json
import re
from loguru import logger

# === Вспомогательные функции ===

def load_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def save_json(obj: Any, p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def build_scene_texts(preproc: dict) -> Dict[Tuple[str, str], dict]:
    """
    Строит словарь: (chapter_id, scene_id) -> { 'text', 'sentences', 'offsets' }
    Использует структуру preproc['chapters'] (список глав).
    """
    texts: Dict[Tuple[str, str], dict] = {}
    for ch in preproc.get('chapters', []):
        ch_id = str(ch.get('id'))
        for sc in ch.get('scenes', []):
            sc_id = str(sc.get('id'))
            text = sc.get('text') or " ".join(s.get('text', '') for s in sc.get('sentences', []))
            sentences = [s.get('text', '') for s in sc.get('sentences', [])]
            offsets: List[Tuple[int, int]] = []
            pos = 0
            for sent in sentences:
                idx = text.find(sent, pos)
                if idx < 0:
                    idx = pos
                offsets.append((idx, idx + len(sent)))
                pos = idx + len(sent)
            texts[(ch_id, sc_id)] = {
                'text': text,
                'sentences': sentences,
                'offsets': offsets
            }
    return texts


def build_mentions_index(chars: Dict[str, dict]) -> Dict[Tuple[str, str], Dict[int, List[str]]]:
    idx: Dict[Tuple[str, str], Dict[int, List[str]]] = {}
    for cid, obj in chars.items():
        for m in obj.get('mentions', []):
            ch = str(m.get('chapter'))
            sc = str(m.get('scene'))
            sid = m.get('sent_id')
            if sid is None:
                continue
            idx.setdefault((ch, sc), {}).setdefault(sid, []).append(cid)
    return idx


def cooc_counts_scene(chars: Dict[str, dict]) -> Dict[str, Dict[str, int]]:
    scene_chars: Dict[Tuple[str, str], set] = {}
    for cid, obj in chars.items():
        for m in obj.get('mentions', []):
            key = (str(m.get('chapter')), str(m.get('scene')))
            scene_chars.setdefault(key, set()).add(cid)
    graph: Dict[str, Dict[str, int]] = {}
    for ids in scene_chars.values():
        for a in ids:
            for b in ids:
                if a == b:
                    continue
                graph.setdefault(a, {}).setdefault(b, 0)
                graph[a][b] += 1
    return graph


def cooc_counts_sent(mentions_idx: Dict[Tuple[str, str], Dict[int, List[str]]]) -> Dict[str, Dict[str, int]]:
    graph: Dict[str, Dict[str, int]] = {}
    for sent_map in mentions_idx.values():
        for ids in sent_map.values():
            unique = set(ids)
            for a in unique:
                for b in unique:
                    if a == b:
                        continue
                    graph.setdefault(a, {}).setdefault(b, 0)
                    graph[a][b] += 1
    return graph


def find_char_by_name(name: str, chars: Dict[str, dict]) -> str | None:
    low = name.lower()
    for cid, data in chars.items():
        if data.get('norm', '').lower() == low or any(low == a.lower() for a in data.get('aliases', [])):
            return cid
    return None

# шаблоны для извлечения отношений через regex
ROLE_WORDS = r"(брат|сестра|жена|муж|сын|дочь|отец|мать|родитель|друг|подруга|коллега|начальник|менеджер|шеф|подчинённый|жених|невеста|товарищ)"
REL_PATTERNS = [
    rf"(?P<a>[А-ЯЁ][а-яё]+(?:\s[А-ЯЁ][а-яё]+)*)\s*[–—-]\s*(?P<label>{ROLE_WORDS})\s+(?P<b>[А-ЯЁ][а-яё]+(?:\s[А-ЯЁ][а-яё]+)*)",
    rf"(?P<a>[А-ЯЁ][а-яё]+(?:\s[А-ЯЁ][а-яё]+)*)\s+(?:это|есть)\s+(?P<label>{ROLE_WORDS})\s+(?P<b>[А-ЯЁ][а-яё]+(?:\s[А-ЯЁ][а-яё]+)*)",
    rf"(?P<label>{ROLE_WORDS})\s+(?P<a>[А-ЯЁ][а-яё]+(?:\s[А-ЯЁ][а-яё]+)*)\s+(?:для|по\sотношению\sк)\s+(?P<b>[А-ЯЁ][а-яё]+(?:\s[А-ЯЁ][а-яё]+)*)",
]


def extract_regex_relations(scene_texts: Dict[Tuple[str, str], dict], chars: Dict[str, dict]) -> List[dict]:
    found: List[dict] = []
    for (ch, sc), obj in scene_texts.items():
        txt = obj['text']
        for pat in REL_PATTERNS:
            for m in re.finditer(pat, txt):
                ca = find_char_by_name(m.group('a'), chars)
                cb = find_char_by_name(m.group('b'), chars)
                if ca and cb:
                    found.append({
                        'a': ca, 'b': cb,
                        'label': m.group('label'),
                        'chapter': ch, 'scene': sc,
                        'span': [m.start(), m.end()],
                        'evidence_text': txt[m.start():m.end()]
                    })
    return found

# === Основная стадия ===

def run_stage(paths: Dict[str, Path], cfg: Dict[str, Any]) -> None:
    book_id = paths['book_root'].name
    coref_path = paths['coref_dir'] / f"{book_id}_coref.json"
    idx_path = paths['coref_dir'] / f"{book_id}_mentions_index.json"
    prep_path = paths['preprocess_dir'] / f"{book_id}_preprocessed.json"

    out_dir = paths['relations_dir']; out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / f"{book_id}_relations_raw.json"
    final_path = out_dir / f"{book_id}_relationships.json"

    if raw_path.exists() and final_path.exists() and not cfg.get('force', False):
        logger.info(f"[relations] Есть результаты, пропускаю: {final_path.name}")
        return

    # Загрузка данных
    chars_data = load_json(coref_path)['entities']
    chars = {str(ent['id']): ent for ent in chars_data}
    mentions_idx = build_mentions_index(chars)
    preproc = load_json(prep_path)

    # Построение промежуточных структур
    scene_texts = build_scene_texts(preproc)
    scene_cooc = cooc_counts_scene(chars)
    sent_cooc = cooc_counts_sent(mentions_idx)
    regex_rel = extract_regex_relations(scene_texts, chars) if cfg.get('regex_roles', True) else []

    # Сохранение «сырых» результатов
    raw = {'cooc_scene': scene_cooc, 'cooc_sent': sent_cooc, 'regex_relations': regex_rel}
    save_json(raw, raw_path)
    logger.info(f"[relations] Raw saved → {raw_path.name}")

    # Постобработка
    scene_thr = cfg.get('scene_min_cooccurs', 2)
    sent_thr = cfg.get('sent_min_cooccurs', 1)
    edges: Dict[str, Dict[str, Any]] = {}

    # cooc_scene
    for a, neigh in scene_cooc.items():
        for b, w in neigh.items():
            if w >= scene_thr:
                edges.setdefault(a, {}).setdefault(b, {'label': 'cooc_scene', 'weight': 0, 'evidence': []})
                edges[a][b]['weight'] = max(edges[a][b]['weight'], w)

    # cooc_sent
    for a, neigh in sent_cooc.items():
        for b, w in neigh.items():
            if w >= sent_thr:
                info = edges.setdefault(a, {}).setdefault(b, {'label': 'cooc_sent', 'weight': 0, 'evidence': []})
                if sent_thr >= scene_thr:
                    edges[a][b]['label'] = 'cooc_sent'
                edges[a][b]['weight'] = max(edges[a][b]['weight'], w)

    # regex_relations
    for r in regex_rel:
        a, b = r['a'], r['b']
        info = edges.setdefault(a, {}).setdefault(b, {'label': r['label'], 'weight': 0, 'evidence': []})
        if cfg.get('regex_roles', True):
            info['label'] = r['label']
        info['weight'] += 1
        info['evidence'].append({'chapter': r['chapter'], 'scene': r['scene'], 'span': r['span'], 'text': r['evidence_text']})

    # Сохранение итоговой структуры
    save_json(edges, final_path)
    logger.info(f"[relations] Final relationships saved → {final_path.name}")
