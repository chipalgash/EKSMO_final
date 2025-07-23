# relationships_extractor.py
from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any
from loguru import logger

# ==== настройки ====
BOOK_ID = "633450"
POSTPROC_DIR = Path("postprocessed")
PREPROC_DIR = Path("preprocessed_texts")
USE_SPACY = False  # опционально (ru_core_news_lg)

# Социальные/родственные роли для регексов
ROLE_WORDS = r"(брат|сестра|жена|муж|сын|дочь|отец|мать|родитель|друг|подруга|коллега|начальник|менеджер|шеф|подчинённый|жених|невеста|товарищ)"
# Шаблоны
REL_PATTERNS = [
    # «А — брат B», «А – жена B»
    rf"(?P<a>[А-ЯЁ][а-яё]+(?:\s[А-ЯЁ][а-яё]+)*)\s*[–—-]\s*(?P<label>{ROLE_WORDS})\s+(?P<b>[А-ЯЁ][а-яё]+(?:\s[А-ЯЁ][а-яё]+)*)",
    # «А это брат B»
    rf"(?P<a>[А-ЯЁ][а-яё]+(?:\s[А-ЯЁ][а-яё]+)*)\s+(?:это|есть)\s+(?P<label>{ROLE_WORDS})\s+(?P<b>[А-ЯЁ][а-яё]+(?:\s[А-ЯЁ][а-яё]+)*)",
    # «Брат А по отношению к B» (обратный порядок)
    rf"(?P<label>{ROLE_WORDS})\s+(?P<a>[А-ЯЁ][а-яё]+(?:\s[А-ЯЁ][а-яё]+)*)\s+(?:для|по отношению к)\s+(?P<b>[А-ЯЁ][а-яё]+(?:\s[А-ЯЁ][а-яё]+)*)",
]

# ==== утилиты ====

def load_json(p: Path) -> Any:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj: Any, p: Path):
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def find_char_by_name(name: str, chars: Dict[str, dict]) -> str | None:
    """Поиск персонажа по норме/алиасу (без учета регистра)."""
    low = name.lower()
    for cid, data in chars.items():
        if data["norm"].lower() == low:
            return cid
        if any(low == a.lower() for a in data["aliases"]):
            return cid
    return None

def build_scene_texts(preproc_book: dict) -> Dict[Tuple[str,str], dict]:
    """map (chapter, scene) -> {text, sentences, offsets}"""
    res = {}
    for ch, scenes in preproc_book.items():
        for sc, data in scenes.items():
            text = data["cleaned_text"]
            sentences = data.get("sentences") or []
            offsets = []
            if sentences:
                # если в препроцессоре не сохранили offsets — посчитаем
                pos = 0
                for s in sentences:
                    idx = text.find(s, pos)
                    offsets.append((idx, idx+len(s)))
                    pos = idx + len(s)
            res[(ch, sc)] = {
                "text": text,
                "sentences": sentences,
                "offsets": offsets
            }
    return res

def build_mentions_index(chars: Dict[str, dict]) -> Dict[Tuple[str,str], Dict[int, List[str]]]:
    """
    Вернём структуру: {(ch,sc): {sent_id: [char_ids...]}}
    чтобы быстро считать ко-упоминания по предложениям.
    """
    idx: Dict[Tuple[str,str], Dict[int, List[str]]] = {}
    for cid, obj in chars.items():
        for m in obj["mentions"]:
            ch = str(m.get("chapter"))
            sc = str(m.get("scene"))
            sid = m.get("sent_id")
            if sid is None:
                continue
            key = (ch, sc)
            idx.setdefault(key, {}).setdefault(sid, []).append(cid)
    return idx

def cooc_counts_scene(chars: Dict[str, dict]) -> Dict[str, Dict[str, int]]:
    """Ко-упоминания по сценам."""
    scene_chars: Dict[Tuple[str,str], set] = {}
    for cid, obj in chars.items():
        for m in obj["mentions"]:
            key = (str(m["chapter"]), str(m["scene"]))
            scene_chars.setdefault(key, set()).add(cid)
    graph: Dict[str, Dict[str, int]] = {}
    for _, ids in scene_chars.items():
        ids = list(ids)
        for i, a in enumerate(ids):
            for b in ids[i+1:]:
                graph.setdefault(a, {}).setdefault(b, 0)
                graph.setdefault(b, {}).setdefault(a, 0)
                graph[a][b] += 1
                graph[b][a] += 1
    return graph

def cooc_counts_sent(mentions_idx: Dict[Tuple[str,str], Dict[int, List[str]]]) -> Dict[str, Dict[str, int]]:
    """Ко-упоминания по предложениям."""
    graph: Dict[str, Dict[str, int]] = {}
    for _, sent_map in mentions_idx.items():
        for _, ids in sent_map.items():
            unique = list(set(ids))
            for i, a in enumerate(unique):
                for b in unique[i+1:]:
                    graph.setdefault(a, {}).setdefault(b, 0)
                    graph.setdefault(b, {}).setdefault(a, 0)
                    graph[a][b] += 1
                    graph[b][a] += 1
    return graph

def extract_regex_relations(scene_texts: Dict[Tuple[str,str], dict],
                            chars: Dict[str, dict]) -> List[dict]:
    """Поиск ролей по заранее заданным регекс-паттернам."""
    found = []
    for (ch, sc), obj in scene_texts.items():
        txt = obj["text"]
        for pat in REL_PATTERNS:
            for m in re.finditer(pat, txt):
                a_name = m.group("a")
                b_name = m.group("b")
                label  = m.group("label")
                ca = find_char_by_name(a_name, chars)
                cb = find_char_by_name(b_name, chars)
                if ca and cb:
                    found.append({
                        "a": ca,
                        "b": cb,
                        "label": label,
                        "chapter": ch,
                        "scene": sc,
                        "span": [m.start(), m.end()],
                        "evidence_text": txt[m.start():m.end()]
                    })
    return found

# ==== основной пайплайн ====

def main(book_id: str = BOOK_ID):
    # загрузка данных
    chars_path = POSTPROC_DIR / f"{book_id}_characters_coref_patched.json"
    if not chars_path.exists():
        # fallback: без патча
        chars_path = POSTPROC_DIR / f"{book_id}_characters_coref.json"
    pre_path = PREPROC_DIR / f"{book_id}_preprocessed.json"

    chars = load_json(chars_path)
    pre = load_json(pre_path)

    # строим карты
    scene_texts = build_scene_texts(pre)
    mentions_idx = build_mentions_index(chars)

    # 1) ко-упоминания
    graph_scene = cooc_counts_scene(chars)
    graph_sent = cooc_counts_sent(mentions_idx)

    # 2) регексы
    regex_edges = extract_regex_relations(scene_texts, chars)

    # 3) (опционально) spaCy  парсинг зависимостной — можно добавить позже
    # if USE_SPACY:
    #     import spacy
    #     nlp = spacy.load("ru_core_news_lg")
    #     # TODO: добавить паттерны с Matcher/DependencyMatcher

    # 4) агрегируем в единый список рёбер
    edges = []
    # ко-упоминания (сцены)
    for a, neigh in graph_scene.items():
        for b, w in neigh.items():
            edge = {
                "a": a, "b": b,
                "label": "cooc_scene",
                "weight": w,
                "evidence": []  # можно позже добавить список сцен
            }
            edges.append(edge)
    # ко-упоминания (предложения)
    for a, neigh in graph_sent.items():
        for b, w in neigh.items():
            edges.append({
                "a": a, "b": b,
                "label": "cooc_sent",
                "weight": w,
                "evidence": []
            })
    # регексы
    for r in regex_edges:
        edges.append({
            "a": r["a"], "b": r["b"],
            "label": r["label"],
            "weight": 1,
            "evidence": [{
                "chapter": r["chapter"],
                "scene": r["scene"],
                "span": r["span"],
                "text": r["evidence_text"]
            }]
        })

    out = {
        "cooc_scene": graph_scene,
        "cooc_sent": graph_sent,
        "regex_relations": regex_edges,
        "edges": edges
    }

    out_path = POSTPROC_DIR / f"{book_id}_relations_raw.json"
    save_json(out, out_path)
    logger.info(f"✓ Relations saved → {out_path}")

if __name__ == "__main__":
    import sys
    bid = sys.argv[1] if len(sys.argv) > 1 else BOOK_ID
    main(bid)
