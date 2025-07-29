# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
NER Validation Stage: проверяет целостность и качество результатов NER.
Чтение:
  30_ner/<book_id>_ner.json
  30_ner/<book_id>_mentions_index.json
Запись:
  30_ner/<book_id>_ner_report.json (опционально)
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
from collections import Counter
from loguru import logger

# -------------------- STAGE --------------------
def run_stage(paths: Dict[str, Path], cfg: Dict[str, Any]) -> None:
    book_id = paths['ner_dir'].name if 'ner_dir' in paths else paths['book_root'].name
    ner_path = paths['ner_dir'] / f"{book_id}_ner.json"
    idx_path = paths['ner_dir'] / f"{book_id}_mentions_index.json"

    # Проверяем наличие файлов
    if not ner_path.exists():
        logger.error(f"[ner_check] Нет файла NER: {ner_path}")
        return
    if not idx_path.exists():
        logger.error(f"[ner_check] Нет индекса упоминаний: {idx_path}")
        return

    # Загружаем данные
    ner_data = json.loads(ner_path.read_text(encoding='utf-8'))
    mentions_index = json.loads(idx_path.read_text(encoding='utf-8'))

    entities = ner_data.get('entities', [])
    total_entities = len(entities)
    total_mentions = 0

    # Собираем все упоминания и проверяем индекс
    missing_in_index: List[Tuple[str,str]] = []
    mention_ids: List[str] = []
    for ent in entities:
        ent_id = ent.get('id')
        mentions = ent.get('mentions', [])
        total_mentions += len(mentions)
        for m in mentions:
            mid = m.get('mention_id')
            mention_ids.append(mid)
            # строим ключ сцены
            ch = m.get('chapter')
            sc = m.get('scene')
            sid = m.get('sent_id')
            key = f"{ch}::{sc}::{sid}"
            # проверяем наличие в mentions_index
            idx_list = mentions_index.get(key, [])
            if mid not in idx_list:
                missing_in_index.append((mid, key))

    # Ищем дубликаты ID
    duplicates = [mid for mid, cnt in Counter(mention_ids).items() if cnt > 1]

    # Статистика по упоминаниям на сущность
    per_entity = [len(ent.get('mentions', [])) for ent in entities]
    avg_mentions = sum(per_entity) / total_entities if total_entities else 0

    # Логируем результаты
    logger.info(f"[ner_check] Всего сущностей: {total_entities}")
    logger.info(f"[ner_check] Всего упоминаний: {total_mentions}")
    logger.info(f"[ner_check] Среднее упоминаний на сущность: {avg_mentions:.2f}")

    if missing_in_index:
        logger.warning(
            f"[ner_check] Упомянено {len(missing_in_index)} случаев, отсутствующих в index.json"  
        )
        # по желанию можно вывести первые 5
        for mid, key in missing_in_index[:5]:
            logger.debug(f"  Missing: {mid} in key {key}")

    if duplicates:
        logger.warning(
            f"[ner_check] Найдены дублирующиеся mention_id: {len(duplicates)} элементов"  
        )
        for mid in duplicates[:5]:
            logger.debug(f"  Duplicate ID: {mid}")

    # По умолчанию сохраняем отчет
    report = {
        'total_entities': total_entities,
        'total_mentions': total_mentions,
        'avg_mentions_per_entity': avg_mentions,
        'missing_in_index': len(missing_in_index),
        'duplicates': len(duplicates),
        'issues_detected': bool(missing_in_index or duplicates)
    }
    report_path = paths['ner_dir'] / f"{book_id}_ner_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    logger.info(f"[ner_check] Report saved → {report_path.name}")



"""
NER-стадия: Natasha + spaCy -> merge -> постобработка (фильтры, кластеризация алиасов)
Сохранение:
  30_ner/<book_id>_ner.json
  30_ner/<book_id>_mentions_index.json
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Tuple, DefaultDict
from collections import defaultdict, Counter
import json

from loguru import logger

# -------------------- Глобальные модели (ленивая загрузка) --------------------

_SPACY_NLP = None
_NATASHA_EXTR = None
_MORPH_VOC = None
_MORPH = None


def _load_spacy(model_name: str = "ru_core_news_lg"):
    global _SPACY_NLP
    if _SPACY_NLP is None:
        import spacy
        try:
            _SPACY_NLP = spacy.load(model_name)
        except OSError:
            logger.error(f"spaCy model '{model_name}' не найдена. Установите:\n  python -m spacy download {model_name}")
            raise
    return _SPACY_NLP


def _load_natasha():
    global _NATASHA_EXTR, _MORPH_VOC
    if _NATASHA_EXTR is None:
        from natasha import MorphVocab, NamesExtractor
        _MORPH_VOC = MorphVocab()
        _NATASHA_EXTR = NamesExtractor(_MORPH_VOC)
    return _NATASHA_EXTR


def _load_morph():
    global _MORPH
    if _MORPH is None:
        try:
            from pymorphy3 import MorphAnalyzer
            _MORPH = MorphAnalyzer()
        except Exception:
            logger.warning("pymorphy3 не установлен, нормализация/пол будут грубыми.")
            _MORPH = None
    return _MORPH


# -------------------- Публичный API --------------------

def run_stage(paths: Dict[str, Path], cfg: Dict[str, Any]) -> None:
    """
    Основной вход для оркестратора.
    """
    book_id = paths["book_root"].name
    in_file = paths["preprocess_dir"] / f"{book_id}_preprocessed.json"  # при необходимости правь имя
    out_dir = paths["ner_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    ner_path = out_dir / f"{book_id}_ner.json"
    idx_path = out_dir / f"{book_id}_mentions_index.json"

    if ner_path.exists() and idx_path.exists() and not cfg.get("force", False):
        logger.info(f"[ner] Уже есть {ner_path.name} и {idx_path.name}, пропускаю.")
        return

    if not in_file.exists():
        raise FileNotFoundError(f"[ner] Нет входного файла: {in_file}")

    logger.info(f"[ner] Читаю {in_file}")
    with in_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # загрузка моделей
    if cfg.get("use_natasha", True):
        _load_natasha()
    if cfg.get("use_spacy", True):
        _load_spacy(cfg.get("spacy_model", "ru_core_news_lg"))
    _load_morph()

    # параметры
    min_len = cfg.get("min_len", 2)
    stopwords = set(cfg.get("stopwords_person_like", []))
    fuzzy_thr = cfg.get("fuzzy_threshold", 90)
    min_mentions = cfg.get("min_mentions", 3)
    min_scenes = cfg.get("min_scenes", 2)
    save_idx = cfg.get("save_mentions_index", True)
    use_morph_gender = cfg.get("use_morph_gender", True)

    # ---- Извлечение -----------------------------------------
    raw_mentions: List[Dict[str, Any]] = []
    for ch in data["chapters"]:
        ch_id = ch["id"]
        for sc in ch["scenes"]:
            sc_id = sc["id"]
            sentences = sc["sentences"]

            scene_text, sent_spans = _concat_sentences(sentences)
            nat_ents = extract_natasha(scene_text) if cfg.get("use_natasha", True) else []
            spa_ents = extract_spacy(scene_text) if cfg.get("use_spacy", True) else []

            merged = merge_entities(nat_ents, spa_ents, min_len=min_len)
            merged = dedup_spans(merged)

            for e in merged:
                e["chapter"] = ch_id
                e["scene"] = sc_id
                e["sent_id"] = _find_sent_id(e["start"], sent_spans)
                e["norm"] = normalize_name(e["text"])
                e["type"] = "name"  # для coref потом будут pronoun и пр.
                raw_mentions.append(e)

    logger.info(f"[ner] Упоминаний до фильтров: {len(raw_mentions)}")

    # ---- Постобработка --------------------------------------
    filtered = [m for m in raw_mentions if not is_noise(m, stopwords)]
    logger.info(f"[ner] После фильтров шума: {len(filtered)}")

    clusters = cluster_by_alias(filtered, fuzzy_thr=fuzzy_thr)
    clusters = apply_frequency_filters(clusters, min_mentions=min_mentions, min_scenes=min_scenes)

    if use_morph_gender:
        for c in clusters:
            c["gender"] = infer_gender(c["aliases"])

    # ---- Проставляем ID -------------------------------------
    global_counter = 0
    mentions_index: DefaultDict[str, List[str]] = defaultdict(list)
    for ent_id, c in enumerate(clusters):
        c["id"] = ent_id
        for m in c["mentions"]:
            m_id = f"ent_{ent_id}:m_{global_counter}"
            m["mention_id"] = m_id
            m["entity_id"] = ent_id
            key = f"{m['chapter']}::{m['scene']}::{m['sent_id']}"
            mentions_index[key].append(m_id)
            global_counter += 1

    # ---- Сохраняем ------------------------------------------
    obj = {"book_id": book_id, "entities": clusters}
    with ner_path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

    if save_idx:
        with idx_path.open("w", encoding="utf-8") as f:
            json.dump(mentions_index, f, ensure_ascii=False, indent=2)

    logger.info(f"[ner] Персонажей: {len(clusters)}")
    logger.info(f"[ner] Сохранено: {ner_path.name}, {idx_path.name}")


# -------------------- Извлечение --------------------

def extract_natasha(text: str) -> List[Dict[str, Any]]:
    extr = _NATASHA_EXTR
    res = []
    for m in extr(text):
        start, stop, span_text = _match_bounds(m, text)
        if start is None:
            continue
        res.append({
            "text": span_text,
            "start": start,
            "end": stop,
            "label": "PER",
            "source": "natasha"
        })
    return res


def extract_spacy(text: str) -> List[Dict[str, Any]]:
    nlp = _SPACY_NLP
    doc = nlp(text)
    res = []
    for ent in doc.ents:
        if ent.label_ == "PER":
            res.append({
                "text": ent.text,
                "start": ent.start_char,
                "end": ent.end_char,
                "label": "PER",
                "source": "spacy"
            })
    return res


# -------------------- Слияние и фильтры --------------------

def merge_entities(nat: List[Dict[str, Any]],
                   spa: List[Dict[str, Any]],
                   min_len: int = 2) -> List[Dict[str, Any]]:
    return [e for e in (nat + spa) if len(e["text"]) >= min_len]


def dedup_spans(ents: List[Dict[str, Any]], iou_thr: float = 0.9) -> List[Dict[str, Any]]:
    kept: List[Dict[str, Any]] = []
    for e in ents:
        drop = False
        for k in kept:
            if span_iou((e["start"], e["end"]), (k["start"], k["end"])) > iou_thr:
                drop = True
                break
        if not drop:
            kept.append(e)
    return kept


def is_noise(m: Dict[str, Any], stopwords: set) -> bool:
    txt = m["text"].strip()
    low = txt.lower()
    # стоп-слова
    if low in stopwords:
        return True
    # одно короткое слово без заглавной
    if len(txt.split()) == 1 and (len(txt) < 2 or not txt[0].isupper()):
        return True
    # POS по первому слову
    morph = _MORPH
    if morph:
        w = txt.split()[0]
        p = morph.parse(w)[0]
        if p.tag.POS in {"VERB", "INFN", "ADJF", "ADVB", "PRCL", "CONJ"}:
            return True
    return False


# -------------------- Кластеризация алиасов --------------------

def cluster_by_alias(mentions: List[Dict[str, Any]], fuzzy_thr: int = 90) -> List[Dict[str, Any]]:
    from rapidfuzz import process, fuzz

    buckets: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
    for m in mentions:
        buckets[m["norm"]].append(m)

    norms = list(buckets.keys())
    used = set()
    clusters: List[Dict[str, Any]] = []

    for n in norms:
        if n in used:
            continue
        cand = [n]
        used.add(n)

        matches = process.extract(n, norms, scorer=fuzz.token_sort_ratio, score_cutoff=fuzzy_thr)
        for other, score, _ in matches:
            if other != n and other not in used:
                cand.append(other)
                used.add(other)

        all_mentions: List[Dict[str, Any]] = []
        aliases_set = set()
        for cn in cand:
            for m in buckets[cn]:
                all_mentions.append(m)
                aliases_set.add(m["text"])

        clusters.append({
            "name": choose_longest_alias(aliases_set),
            "aliases": sorted(aliases_set),
            "norms": cand,
            "mentions": sorted(all_mentions, key=lambda x: (x["chapter"], x["scene"], x["sent_id"], x["start"]))
        })

    return clusters


def apply_frequency_filters(clusters: List[Dict[str, Any]],
                            min_mentions: int,
                            min_scenes: int) -> List[Dict[str, Any]]:
    kept = []
    for c in clusters:
        mcnt = len(c["mentions"])
        scns = len({(m["chapter"], m["scene"]) for m in c["mentions"]})
        if mcnt >= min_mentions and scns >= min_scenes:
            kept.append(c)
    return kept


# -------------------- Вспомогательные --------------------

def _concat_sentences(sentences: List[Dict[str, Any]]) -> Tuple[str, List[Tuple[int, int, int]]]:
    chunks = []
    spans: List[Tuple[int, int, int]] = []
    pos = 0
    for s in sentences:
        txt = s["text"]
        chunks.append(txt)
        start = pos
        end = pos + len(txt)
        spans.append((s["id"], start, end))
        pos = end + 1
    return " ".join(chunks), spans


def _find_sent_id(char_pos: int, sent_spans: List[Tuple[int, int, int]]) -> int:
    for sid, st, en in sent_spans:
        if st <= char_pos < en:
            return sid
    return -1


def _match_bounds(m, text: str):
    if hasattr(m, "span"):
        try:
            return m.span.start, m.span.stop, m.span.text
        except Exception:
            pass
    if hasattr(m, "start") and hasattr(m, "stop"):
        start, stop = m.start, m.stop
        return start, stop, text[start:stop]
    return None, None, None


def span_iou(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union else 0.0


def choose_longest_alias(aliases: set[str]) -> str:
    return max(aliases, key=len)


def normalize_name(text: str) -> str:
    morph = _MORPH
    parts = [p for p in text.strip().split() if p]
    if not morph:
        return " ".join(p.lower() for p in parts)
    normed = []
    for w in parts:
        parsed = morph.parse(w)
        normed.append(parsed[0].normal_form if parsed else w.lower())
    return " ".join(normed)


def infer_gender(aliases: List[str]) -> str | None:
    """
    Эвристика: считаем по всем алиасам.
    1) pymorphy.gender у первого слова
    2) частота masc/femn > 1
    3) fallback по окончаниям (-а/-я/-ия -> femn)
    """
    morph = _MORPH
    if not morph or not aliases:
        return None

    counts = Counter()
    for a in aliases:
        first = a.split()[0]
        p = morph.parse(first)[0]
        if p.tag.gender in {"masc", "femn"}:
            counts[p.tag.gender] += 1

    if counts:
        return "male" if counts["masc"] >= counts["femn"] else "female"

    # fallback по окончаниям
    for a in aliases:
        w = a.split()[0].lower()
        if w.endswith(("а", "я", "ия")):
            return "female"
    return None
