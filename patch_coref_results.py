# patch_coref_results.py
from __future__ import annotations
import json
from pathlib import Path
from loguru import logger

try:
    from pymorphy3 import MorphAnalyzer
except ImportError:
    from pymorphy2 import MorphAnalyzer

morph = MorphAnalyzer()

PLURAL_PRONOUNS = {"они", "их", "им", "ими", "них"}  # будем выкидывать
BOOK_ID = "633450"  # поменяй при необходимости

def detect_gender_by_name(name: str) -> str:
    tokens = name.split()
    for tok in reversed(tokens):
        p = morph.parse(tok)[0]
        g = getattr(p.tag, "gender", None)
        if g == "masc":
            return "male"
        if g == "femn":
            return "female"
    return "unknown"

def patch():
    chars_path = Path("postprocessed") / f"{BOOK_ID}_characters_coref.json"
    added_path = Path("postprocessed") / f"{BOOK_ID}_coref_mentions.json"

    if not chars_path.exists():
        logger.error(f"Нет файла {chars_path}")
        return

    chars = json.loads(chars_path.read_text(encoding="utf-8"))
    added = []
    if added_path.exists():
        added = json.loads(added_path.read_text(encoding="utf-8"))

    # 1. gender fix
    fixed_g = 0
    for cid, obj in chars.items():
        g = obj.get("gender")
        if not g or g == "unknown":
            new_g = detect_gender_by_name(obj.get("norm", ""))
            if new_g != "unknown":
                obj["gender"] = new_g
                fixed_g += 1

    # 2. filter plural pronouns + 3. ensure type
    removed_plural = 0
    for cid, obj in chars.items():
        new_mentions = []
        for m in obj.get("mentions", []):
            # type
            if "type" not in m:
                # если слово длиннее 1 и не в PRONOUNS — пусть будет name
                m["type"] = "pronoun" if m.get("text", "").lower() in PLURAL_PRONOUNS else "name"
            # фильтрация plurals
            if m["type"] == "pronoun" and m.get("text", "").lower() in PLURAL_PRONOUNS:
                removed_plural += 1
                continue
            new_mentions.append(m)
        obj["mentions"] = new_mentions

    # Также чистим файл coref_mentions.json, если он есть
    new_added = []
    for m in added:
        if m.get("type") == "pronoun" and m.get("text", "").lower() in PLURAL_PRONOUNS:
            removed_plural += 1
            continue
        if "type" not in m:
            m["type"] = "pronoun" if m.get("text", "").lower() in PLURAL_PRONOUNS else "name"
        new_added.append(m)

    # сохраняем
    out_chars = Path("postprocessed") / f"{BOOK_ID}_characters_coref_patched.json"
    out_added = Path("postprocessed") / f"{BOOK_ID}_coref_mentions_patched.json"
    out_chars.write_text(json.dumps(chars, ensure_ascii=False, indent=2), encoding="utf-8")
    if added_path.exists():
        out_added.write_text(json.dumps(new_added, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info(f"✓ genders fixed: {fixed_g}")
    logger.info(f"✓ plural pronouns removed: {removed_plural}")
    logger.info(f"→ {out_chars}")
    if added_path.exists():
        logger.info(f"→ {out_added}")

if __name__ == "__main__":
    patch()
