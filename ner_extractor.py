# -*- coding: utf-8 -*-
"""
NER Stage: Natasha + spaCy → merge → post‐processing (кластеризация, фильтры) → сохранение ner.json и mentions_index.json

Чтение:
  workspace/.../20_preprocessed/<book_id>_preprocessed.json

Запись:
  workspace/.../30_ner/<book_id>_ner.json
  workspace/.../30_ner/<book_id>_mentions_index.json
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, DefaultDict
from collections import defaultdict, Counter
from loguru import logger

# ——— ЛЕНИВАЯ ЗАГРУЗКА МОДЕЛЕЙ —————————————————————————
_SPACY_NLP    = None
_NATASHA_EXTR = None
_MORPH_VOC    = None
_MORPH        = None

def _load_spacy(model_name: str = "ru_core_news_lg"):
    global _SPACY_NLP
    if _SPACY_NLP is None:
        import spacy
        try:
            _SPACY_NLP = spacy.load(model_name)
        except OSError:
            logger.error(f"spaCy model '{model_name}' не найдена. Установите: python -m spacy download {model_name}")
            raise
    return _SPACY_NLP

def _load_natasha():
    global _NATASHA_EXTR, _MORPH_VOC
    if _NATASHA_EXTR is None:
        from natasha import MorphVocab, NamesExtractor
        _MORPH_VOC    = MorphVocab()
        _NATASHA_EXTR = NamesExtractor(_MORPH_VOC)
    return _NATASHA_EXTR

def _load_morph():
    global _MORPH
    if _MORPH is None:
        try:
            from pymorphy3 import MorphAnalyzer
            _MORPH = MorphAnalyzer()
        except ImportError:
            logger.warning("pymorphy3 не установлен — нормализация будет грубой.")
            _MORPH = None
    return _MORPH

# ——— ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ——————————————————————————

def _concat_sentences(sentences: List[Dict[str,Any]]) -> Tuple[str, List[Tuple[int,int,int]]]:
    parts: List[str] = []
    spans: List[Tuple[int,int,int]] = []
    pos = 0
    for s in sentences:
        text = s["text"]
        parts.append(text)
        start = pos
        end   = pos + len(text)
        spans.append((s["id"], start, end))
        pos = end + 1
    return " ".join(parts), spans

def _find_sent_id(char_pos: int, sent_spans: List[Tuple[int,int,int]]) -> int:
    for sid, st, en in sent_spans:
        if st <= char_pos < en:
            return sid
    return -1

def _match_bounds(matcher, text: str) -> Tuple[int,int,str]:
    if hasattr(matcher, "span"):
        try:
            return matcher.span.start, matcher.span.stop, matcher.span.text
        except Exception:
            pass
    if hasattr(matcher, "start") and hasattr(matcher, "stop"):
        st, en = matcher.start, matcher.stop
        return st, en, text[st:en]
    return None, None, None

def extract_natasha(text: str) -> List[Dict[str,Any]]:
    extr = _load_natasha()
    res: List[Dict[str,Any]] = []
    for m in extr(text):
        st, en, span = _match_bounds(m, text)
        if st is None:
            continue
        res.append({
            "text":   span,
            "start":  st,
            "end":    en,
            "label":  "PER",
            "source": "natasha"
        })
    return res

def extract_spacy(text: str) -> List[Dict[str,Any]]:
    nlp = _load_spacy()
    doc = nlp(text)
    res: List[Dict[str,Any]] = []
    for ent in doc.ents:
        if ent.label_ == "PER":
            res.append({
                "text":   ent.text,
                "start":  ent.start_char,
                "end":    ent.end_char,
                "label":  "PER",
                "source": "spacy"
            })
    return res

def dedup_spans(ents: List[Dict[str,Any]], iou_thr: float = 0.9) -> List[Dict[str,Any]]:
    kept: List[Dict[str,Any]] = []
    for e in ents:
        a = (e["start"], e["end"])
        drop = False
        for k in kept:
            b = (k["start"], k["end"])
            inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
            union = (a[1]-a[0]) + (b[1]-b[0]) - inter
            if union and (inter/union) > iou_thr:
                drop = True
                break
        if not drop:
            kept.append(e)
    return kept

def normalize_name(text: str) -> str:
    morph = _load_morph()
    parts = [w for w in text.split() if w]
    if not morph:
        return " ".join(p.lower() for p in parts)
    normed: List[str] = []
    for w in parts:
        p = morph.parse(w)[0]
        normed.append(p.normal_form)
    return " ".join(normed)

def is_noise(m: Dict[str,Any], stopwords: set) -> bool:
    txt = m["text"].strip()
    low = txt.lower()
    if low in stopwords:
        return True
    if len(txt.split()) == 1 and (len(txt) < 2 or not txt[0].isupper()):
        return True
    morph = _load_morph()
    if morph:
        w = txt.split()[0]
        p = morph.parse(w)[0]
        if p.tag.POS in {"VERB","INFN","ADJF","ADVB","PRCL","CONJ"}:
            return True
    return False

def cluster_by_alias(
    mentions: List[Dict[str,Any]],
    fuzzy_thr: int = 90
) -> List[Dict[str,Any]]:
    from rapidfuzz import process, fuzz

    buckets: DefaultDict[str, List[Dict[str,Any]]] = defaultdict(list)
    for m in mentions:
        buckets[m["norm"]].append(m)

    norms = list(buckets.keys())
    used = set()
    clusters: List[Dict[str,Any]] = []

    for n in norms:
        if n in used:
            continue
        group = [n]
        used.add(n)
        matches = process.extract(n, norms,
                                  scorer=fuzz.token_sort_ratio,
                                  score_cutoff=fuzzy_thr)
        for other, score, _ in matches:
            if other not in used:
                group.append(other)
                used.add(other)

        all_m: List[Dict[str,Any]] = []
        aliases = set()
        for nm in group:
            for m in buckets[nm]:
                all_m.append(m)
                aliases.add(m["text"])

        clusters.append({
            "name":     max(aliases, key=len),
            "aliases":  sorted(aliases),
            "norms":    group,
            "mentions": sorted(all_m, key=lambda x: (
                             x["chapter"], x["scene"],
                             x["sent_id"], x["start"]))
        })

    return clusters

def apply_frequency_filters(
    clusters: List[Dict[str,Any]],
    min_mentions: int,
    min_scenes:   int
) -> List[Dict[str,Any]]:
    kept: List[Dict[str,Any]] = []
    for c in clusters:
        mcnt = len(c["mentions"])
        scns = len({(m["chapter"], m["scene"]) for m in c["mentions"]})
        if mcnt >= min_mentions and scns >= min_scenes:
            kept.append(c)
    return kept

def infer_gender(aliases: List[str]) -> str | None:
    morph = _load_morph()
    if not morph or not aliases:
        return None
    counts = Counter()
    for a in aliases:
        first = a.split()[0]
        p = morph.parse(first)[0]
        if p.tag.gender in {"masc","femn"}:
            counts[p.tag.gender] += 1
    if counts:
        return "male" if counts["masc"] >= counts["femn"] else "female"
    for a in aliases:
        w = a.split()[0].lower()
        if w.endswith(("а","я","ия")):
            return "female"
    return None

# ——— ОСНОВНАЯ ФУНКЦИЯ ————————————————————————————————————

def run_stage(paths: Dict[str, Path], cfg: Dict[str,Any]) -> None:
    book_id    = paths["book_root"].name
    in_file    = paths["preprocess_dir"] / f"{book_id}_preprocessed.json"
    ner_path   = paths["ner_dir"]        / f"{book_id}_ner.json"
    idx_path   = paths["ner_dir"]        / f"{book_id}_mentions_index.json"

    paths["ner_dir"].mkdir(parents=True, exist_ok=True)

    if ner_path.exists() and idx_path.exists() and not cfg.get("force", False):
        logger.info(f"[ner] {ner_path.name} и {idx_path.name} уже есть — пропускаем.")
        return

    if not in_file.exists():
        raise FileNotFoundError(f"[ner] Не найден входной файл: {in_file}")

    data = json.loads(in_file.read_text(encoding="utf-8"))

    if cfg.get("use_spacy", True):
        _load_spacy(cfg.get("spacy_model", "ru_core_news_lg"))
    if cfg.get("use_natasha", True):
        _load_natasha()
    _load_morph()

    min_len      = cfg.get("min_len", 2)
    stopwords    = set(cfg.get("stopwords_person_like", []))
    fuzzy_thr    = cfg.get("fuzzy_threshold", 90)
    min_mentions = cfg.get("min_mentions", 3)
    min_scenes   = cfg.get("min_scenes", 2)

    # 1) Извлечение «сырых» упоминаний
    raw_mentions: List[Dict[str,Any]] = []
    for ch in data["chapters"]:
        for sc in ch["scenes"]:
            text, spans = _concat_sentences(sc["sentences"])
            nat = extract_natasha(text) if cfg.get("use_natasha", True) else []
            spa = extract_spacy(text)   if cfg.get("use_spacy", True)   else []
            merged = dedup_spans([e for e in (nat + spa) if len(e["text"]) >= min_len])
            for e in merged:
                e.update({
                    "chapter":  ch["id"],
                    "scene":    sc["id"],
                    "sent_id":  _find_sent_id(e["start"], spans),
                    "norm":     normalize_name(e["text"]),
                    "type":     "name",
                })
            raw_mentions.extend(merged)

    logger.info(f"[ner] Упоминаний до фильтрации: {len(raw_mentions)}")

    # 2) Фильтрация шума
    filtered = [m for m in raw_mentions if not is_noise(m, stopwords)]
    logger.info(f"[ner] После фильтра шума: {len(filtered)}")

    # 3) Кластеризация и частотные фильтры
    clusters = cluster_by_alias(filtered, fuzzy_thr)
    clusters = apply_frequency_filters(clusters, min_mentions, min_scenes)
    if cfg.get("use_morph_gender", True):
        for c in clusters:
            c["gender"] = infer_gender(c["aliases"])

    # 4) Присвоение ID и построение index
    global_counter = 0
    mentions_index: DefaultDict[str, List[str]] = defaultdict(list)
    for ent_id, c in enumerate(clusters):
        c["id"] = ent_id
        for m in c["mentions"]:
            m_id = f"ent_{ent_id}:m_{global_counter}"
            m["mention_id"] = m_id
            m["entity_id"]  = ent_id
            key = f"{m['chapter']}::{m['scene']}::{m['sent_id']}"
            mentions_index[key].append(m_id)
            global_counter += 1

    # 5) Сохранение
    with ner_path.open("w", encoding="utf-8") as f:
        json.dump({"book_id": book_id, "entities": clusters}, f, ensure_ascii=False, indent=2)
    with idx_path.open("w", encoding="utf-8") as f:
        json.dump(mentions_index, f, ensure_ascii=False, indent=2)

    logger.info(f"[ner] Персонажей: {len(clusters)}; сохранено → "
                f"{ner_path.name}, {idx_path.name}")
