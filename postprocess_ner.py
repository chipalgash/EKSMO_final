from __future__ import annotations
import json
import re
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Any

from loguru import logger

# ---- Morph analyzer ----
try:
    from pymorphy3 import MorphAnalyzer  # python 3.11 friendly
except ImportError:  # fallback, но лучше ставить pymorphy3
    from pymorphy2 import MorphAnalyzer

morph = MorphAnalyzer()

# -------- параметры фильтрации --------
RUS_LETTERS = re.compile(r"[а-яА-ЯёЁ]")
BAD_PUNCT_RE = re.compile(r"[\"“”«»\[\]{}<>]|(--+)|(^[-–—])|([?!]{2,})")
DIALOGUE_MARK_RE = re.compile(r"\s?[–—-]\s?")  # для реплик "Угу — Рей"
APOSTROPHES = "[’'`´ʼ]"

STOP_TOKENS = {
    "глава", "часть", "сцена", "эпизод",
    "он", "она", "они", "ему", "ей", "их", "ее", "его",
    "свой", "твой", "мой", "наш", "ваш",
    "что", "это", "который", "которые", "которую",
    "кто", "как", "здесь", "там", "тогда", "сейчас"
}

MIN_ALIAS_LEN = 3      # минимальная длина алиаса
MIN_CHAR_FREQ = 3      # минимум упоминаний персонажа, чтобы не считать шумом
GOOD_POS = {"NOUN", "NPRO", "NAME", "SURN"}  # плюс Name/Surn в тэге
CASE_NOM = {'nomn'}


def detect_gender_by_name(name: str) -> str:
    # Берём последнее слово (имя/фамилия) и смотрим морфологию
    last = name.split()[-1]
    p = morph.parse(last)[0]
    if 'masc' in p.tag.gender:
        return 'male'
    if 'femn' in p.tag.gender:
        return 'female'
    return 'unknown'

# ----------------- helpers -----------------


def word_to_nom(token: str) -> str:
    p = morph.parse(token)[0]
    # если уже в именительном или это имя собственное
    if getattr(p.tag, "case", None) in CASE_NOM or 'Name' in p.tag or 'Surn' in p.tag:
        base = p.normal_form
    else:
        cand = p.inflect(CASE_NOM)
        base = cand.word if cand else p.normal_form
    return base.capitalize()


def to_nominative(s: str) -> str:
    toks = re.findall(r"\w+", s, flags=re.UNICODE)
    if not toks:
        return s
    return " ".join(word_to_nom(t) for t in toks)


def clean_alias(alias: str) -> str:
    a = alias.strip()
    # выбрасываем короткие диалоговые реплики вида «Угу – Рей» и т.п.
    if DIALOGUE_MARK_RE.search(a) and len(a.split()) <= 3:
        return ""
    a = BAD_PUNCT_RE.sub(" ", a)
    a = re.sub(r"\s+", " ", a)
    return a.strip()


def is_noise(name: str) -> bool:
    if len(name) < MIN_ALIAS_LEN:
        return True
    if not RUS_LETTERS.search(name):
        return True
    tokens = re.findall(r"\w+", name.lower())
    if not tokens:
        return True
    good = 0
    for t in tokens:
        if t in STOP_TOKENS:
            continue
        p = morph.parse(t)[0]
        if p.tag.POS in GOOD_POS or 'Name' in p.tag or 'Surn' in p.tag:
            good += 1
    return good == 0


def canonical_key(name: str) -> str:
    name_wo_apos = re.sub(APOSTROPHES, "", name)
    tokens = re.findall(r"\w+", name_wo_apos.lower())
    lemmas = [morph.parse(t)[0].normal_form for t in tokens]
    return " ".join(lemmas)


def choose_best_norm(aliases: List[str]) -> str:
    if not aliases:
        return ""
    best = max(aliases, key=len)
    return to_nominative(best)


def make_char_id(norm: str) -> str:
    h = hashlib.md5(norm.encode("utf-8")).hexdigest()[:8]
    return f"char_{h}"


def json_default(obj: Any):
    if isinstance(obj, set):
        return list(obj)
    return str(obj)


def normalize_mention(m, chapter, scene):
    """Приводит mention к dict-формату: {chapter, scene, start, end, text}."""
    if isinstance(m, dict):
        return {
            "chapter": m.get("chapter", chapter),
            "scene":   m.get("scene", scene),
            "start":   m.get("start"),
            "end":     m.get("end"),
            "text":    m.get("text", "")
        }
    if isinstance(m, (list, tuple)):
        if len(m) == 3:
            start, end, text = m
            return {"chapter": chapter, "scene": scene, "start": start, "end": end, "text": text}
        if len(m) == 4:
            # либо (start,end,text,extra), либо (chapter,scene,start,end)
            if isinstance(m[0], (int, float)) and isinstance(m[1], (int, float)):
                start, end, text, _ = m
                return {"chapter": chapter, "scene": scene, "start": start, "end": end, "text": text}
            ch, sc, start, end = m
            return {"chapter": ch, "scene": sc, "start": start, "end": end, "text": ""}
        if len(m) == 5:
            ch, sc, start, end, text = m
            return {"chapter": ch, "scene": sc, "start": start, "end": end, "text": text}
    # fallback
    return {"chapter": chapter, "scene": scene, "start": None, "end": None, "text": str(m)}


def detect_gender_by_name(name: str) -> str:
    # Берём последнее слово (имя/фамилия) и смотрим морфологию
    last = name.split()[-1]
    p = morph.parse(last)[0]
    if 'masc' in p.tag.gender:
        return 'male'
    if 'femn' in p.tag.gender:
        return 'female'
    return 'unknown'

# ----------------- загрузка и мердж -----------------


def load_ner_files(in_dir: Path) -> List[Tuple[Path, dict]]:
    files = []
    for p in in_dir.glob("*.json"):
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        files.append((p, data))
    return files


def merge_book_persons(book_ner: dict) -> Dict[str, dict]:
    """
    book_ner: {chapter: {scene: {canon_key: {aliases, mentions, norm}}}}
    -> глобальный словарь по ключу канонизации
    """
    global_chars: Dict[str, dict] = {}

    for chapter, scenes in book_ner.items():
        for scene, chars in scenes.items():
            for _key, info in chars.items():
                aliases = info.get("aliases") or []
                if isinstance(aliases, set):
                    aliases = list(aliases)
                mentions = info.get("mentions") or []
                norm = info.get("norm") or choose_best_norm(aliases)

                # чистим алиасы
                cleaned_aliases = []
                for a in aliases:
                    ca = clean_alias(a)
                    if ca and not is_noise(ca):
                        cleaned_aliases.append(ca)

                # решаем, есть ли что нормализовать
                if not cleaned_aliases and is_noise(norm):
                    continue

                if not norm or is_noise(norm):
                    norm = choose_best_norm(cleaned_aliases) if cleaned_aliases else norm
                if not norm:
                    continue

                norm = to_nominative(norm)
                new_key = canonical_key(norm)

                bucket = global_chars.setdefault(new_key, {
                    "aliases": [],
                    "mentions": [],
                    "norm": norm
                })

                for a in cleaned_aliases:
                    if a not in bucket["aliases"]:
                        bucket["aliases"].append(a)

                for m in mentions:
                    mn = normalize_mention(m, chapter, scene)
                    # фильтр диалоговых кусков
                    if DIALOGUE_MARK_RE.search(mn["text"]) and len(mn["text"].split()) <= 3:
                        continue
                    bucket["mentions"].append(mn)

                if len(norm) > len(bucket["norm"]):
                    bucket["norm"] = norm

    # фильтрация по частоте и финальная чистка
    cleaned = {}
    for k, v in global_chars.items():
        freq = len(v["mentions"])
        if freq < MIN_CHAR_FREQ:
            continue
        v["aliases"] = sorted(set(a for a in v["aliases"] if not is_noise(a)))
        cleaned[k] = v

    return cleaned


def build_indices(characters: Dict[str, dict], gender=None) -> Tuple[Dict[str, dict], Dict[str, str]]:
    """
    Создаём char_id и индекс упоминаний.
    returns:
      characters_with_ids: {char_id: {...}}
      mentions_index: {mention_id: char_id}
    """
    char_by_id = {}
    mention_index = {}

    for canon_key, data in characters.items():
        norm = data["norm"]
        char_id = make_char_id(norm)
        aliases = data["aliases"]
        mentions = data["mentions"]

        char_obj = {
            "id": char_id,
            "norm": norm,
            "gender": gender,
            "aliases": aliases,
            "mentions": []
        }

        for i, m in enumerate(mentions, start=1):
            mention_id = f"{char_id}_m{i}"
            m["id"] = mention_id
            char_obj["mentions"].append(m)
            mention_index[mention_id] = char_id

        char_by_id[char_id] = char_obj

    return char_by_id, mention_index

# ----------------- main -----------------


def postprocess_one_book(path_in: Path, out_dir: Path):
    with path_in.open("r", encoding="utf-8") as f:
        book_ner = json.load(f)

    merged = merge_book_persons(book_ner)
    chars_with_ids, mention_idx = build_indices(merged)

    out_dir.mkdir(exist_ok=True)
    base = path_in.stem.replace("_preprocessed_ner", "")

    chars_file = out_dir / f"{base}_characters_clean.json"
    idx_file = out_dir / f"{base}_mentions_index.json"

    with chars_file.open("w", encoding="utf-8") as f:
        json.dump(chars_with_ids, f, ensure_ascii=False, indent=2, default=json_default)

    with idx_file.open("w", encoding="utf-8") as f:
        json.dump(mention_idx, f, ensure_ascii=False, indent=2)

    logger.info(f"✓ Postprocess: {chars_file}")
    logger.info(f"✓ Mentions index: {idx_file}")


if __name__ == "__main__":
    INPUT_DIR = Path("ner_outputs")
    OUTPUT_DIR = Path("postprocessed")

    files = load_ner_files(INPUT_DIR)
    if not files:
        logger.warning("Нет файлов в ner_outputs/*.json")
    for p, _ in files:
        logger.info(f"Postprocess: {p.name}")
        postprocess_one_book(p, OUTPUT_DIR)
