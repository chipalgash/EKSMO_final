from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
from loguru import logger
import re

def load_json(p: Path) -> Any:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj: Any, p: Path):
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def build_sentence_offsets(text: str, sentences: List[str]) -> List[Tuple[int, int]]:
    """
    Возвращает список (start, end) для каждого предложения в тексте.
    Релятивный поиск с учётом повтора предложений.
    """
    offsets = []
    pos = 0
    for s in sentences:
        idx = text.find(s, pos)
        if idx == -1:
            # fallback: грубый поиск через регэксп
            m = re.search(re.escape(s[:20]), text[pos:])
            if m:
                idx = pos + m.start()
            else:
                idx = pos
        offsets.append((idx, idx + len(s)))
        pos = idx + len(s)
    return offsets

def enrich_mentions(book_id: str,
                    preproc_dir: Path = Path("preprocessed_texts"),
                    postproc_dir: Path = Path("postprocessed")):
    preproc_path = preproc_dir / f"{book_id}_preprocessed.json"
    chars_path = postproc_dir / f"{book_id}_characters_clean.json"

    preproc = load_json(preproc_path)
    chars = load_json(chars_path)

    # Соберём карты сцен -> (text, sentences, offsets)
    scene_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for chapter, scenes in preproc.items():
        for scene, data in scenes.items():
            text = data["cleaned_text"]
            sentences = data.get("sentences") or []
            if not sentences:
                # fallback: грубая токенизация
                sentences = re.split(r'(?<=[.!?…])\\s+', text)
                sentences = [s.strip() for s in sentences if s.strip()]
            offsets = build_sentence_offsets(text, sentences)
            scene_cache[(chapter, scene)] = {
                "text": text,
                "sentences": sentences,
                "offsets": offsets
            }

    # Обогащаем mentions
    updated = 0
    for cid, obj in chars.items():
        for m in obj["mentions"]:
            ch = str(m.get("chapter"))
            sc = str(m.get("scene"))
            key = (ch, sc)
            scene_info = scene_cache.get(key)
            if not scene_info:
                continue
            start = m.get("start")
            if start is None:
                continue

            # найти предложение
            sent_id = None
            for idx, (s_start, s_end) in enumerate(scene_info["offsets"]):
                if s_start <= start < s_end:
                    sent_id = idx
                    break
            if sent_id is None:
                continue

            sent_start = scene_info["offsets"][sent_id][0]
            rel_start = start - sent_start
            rel_end = m.get("end", start) - sent_start

            m["sent_id"] = sent_id
            m["start_in_sent"] = rel_start
            m["end_in_sent"] = rel_end
            updated += 1

    out_path = postproc_dir / f"{book_id}_characters_clean_enriched.json"
    save_json(chars, out_path)
    logger.info(f"✓ Enriched mentions ({updated}) → {out_path}")

if __name__ == "__main__":

    import sys
    book = sys.argv[1] if len(sys.argv) > 1 else "633450"
    enrich_mentions(book)
