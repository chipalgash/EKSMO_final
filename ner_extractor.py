# ner_extractor.py
from __future__ import annotations
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import spacy
from natasha import (
    Segmenter, Doc, NewsEmbedding, NewsNERTagger,
    MorphVocab, NamesExtractor
)

from loguru import logger
from pymorphy3 import MorphAnalyzer

# --------- инициализация моделей ----------
segmenter = Segmenter()
emb = NewsEmbedding()
natasha_ner = NewsNERTagger(emb)
morph_vocab = MorphVocab()
names_extractor = NamesExtractor(morph_vocab)

morph = MorphAnalyzer()


try:
    from pymorphy3 import MorphAnalyzer
except ImportError:
    from pymorphy3 import MorphAnalyzer  # на всякий случай


try:
    nlp_spacy = spacy.load("ru_core_news_lg")
except OSError:
    logger.error("Модель spaCy 'ru_core_news_lg' не найдена. Установите: python -m spacy download ru_core_news_lg")
    raise

# --------- служебные функции ----------

def run_natasha(text: str) -> List[Dict]:
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_ner(natasha_ner)

    ents = []
    for span in doc.spans:
        if span.type == 'PER':  # только персонажи
            span.normalize(morph_vocab)
            ents.append({
                "text": span.text,
                "start": span.start,
                "end": span.stop,
                "label": "PER",
                "norm": span.normal
            })
    return ents

def run_spacy(text: str) -> List[Dict]:
    doc = nlp_spacy(text)
    ents = []
    for ent in doc.ents:
        if ent.label_ == "PER":
            ents.append({
                "text": ent.text,
                "start": ent.start_char,
                "end": ent.end_char,
                "label": "PER",
            })
    return ents

def iou(span1: Tuple[int, int], span2: Tuple[int, int]) -> float:
    a0, a1 = span1
    b0, b1 = span2
    inter = max(0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union else 0

def merge_entities(natasha_ents, spacy_ents, iou_thr: float = 0.5):
    merged = natasha_ents[:]
    for s_ent in spacy_ents:
        s_span = (s_ent["start"], s_ent["end"])
        if not any(iou(s_span, (m["start"], m["end"])) >= iou_thr for m in merged):
            s_ent["norm"] = try_normalize_name(s_ent["text"])
            merged.append(s_ent)
    return merged

def try_normalize_name(name: str) -> str:
    """
    Возвращает нормализованное ФИО: Имя Отчество Фамилия (если удалось),
    иначе — капитализированный оригинал.
    """
    match = names_extractor.find(name)
    if match:
        fact = match.fact  # fact.first, fact.middle, fact.last
        parts = [fact.first, fact.middle, fact.last]
        parts = [p for p in parts if p]
        # лемматизируем pymorphy2, затем приводим к Title Case
        norm_parts = [morph.parse(p)[0].normal_form.capitalize() for p in parts]
        return " ".join(norm_parts)

    # fallback: просто капитализация каждого слова
    tokens = re.findall(r"\w+", name, flags=re.UNICODE)
    return " ".join(t.capitalize() for t in tokens)

def canonical_key(name: str) -> str:
    """Ключ для группировки алиасов (нормализация + лемматизация слов)."""
    tokens = re.findall(r"\w+", name.lower())
    lemmas = [morph.parse(t)[0].normal_form for t in tokens]
    return " ".join(lemmas)

def deduplicate_persons(ents: List[Dict]) -> Dict[str, Dict]:
    """
    Группируем сущности по каноническому ключу. Возвращаем структуру:
    {canon_key: {"aliases": set(), "mentions": [(start,end,text)], "norm": best_norm}}
    """
    persons = {}
    for e in ents:
        norm = e.get("norm") or try_normalize_name(e["text"])
        key = canonical_key(norm)
        bucket = persons.setdefault(key, {"aliases": set(), "mentions": [], "norm": norm})
        bucket["aliases"].add(e["text"])
        bucket["mentions"].append((e["start"], e["end"], e["text"]))
        # возможно обновление norm (например, выбрать максимальной длины)
        if len(norm) > len(bucket["norm"]):
            bucket["norm"] = norm
    return persons

# --------- основной пайплайн ----------

def process_scene_text(text: str) -> Dict:
    """
    Возвращает структуру с персонажами для одной сцены.
    """
    nat_ents = run_natasha(text)
    sp_ents = run_spacy(text)
    merged = merge_entities(nat_ents, sp_ents)

    grouped = deduplicate_persons(merged)
    return grouped


def json_default(obj):
    if isinstance(obj, set):
        return list(obj)
    return obj


def process_book_preprocessed(json_path: Path, output_dir: Path):
    with json_path.open("r", encoding="utf-8") as f:
        book = json.load(f)

    result = {}
    for chapter, scenes in book.items():
        result[chapter] = {}
        for scene, data in scenes.items():
            scene_text = data["cleaned_text"]  # можно брать cleaned_text или исходный
            chars = process_scene_text(scene_text)
            result[chapter][scene] = chars

    output_dir.mkdir(exist_ok=True)
    out_file = output_dir / f"{json_path.stem}_ner.json"
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=json_default)
    logger.info(f"NER-результат сохранён в {out_file}")

if __name__ == "__main__":
    preproc_dir = Path("preprocessed_texts")
    ner_out = Path("ner_outputs")

    for p in preproc_dir.glob("*_preprocessed.json"):
        logger.info(f"NER для {p}")
        process_book_preprocessed(p, ner_out)
