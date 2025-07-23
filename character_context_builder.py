# character_context_builder.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
from loguru import logger
import re

BOOK_ID = "633450"
PREPROC_DIR = Path("preprocessed_texts")
POSTPROC_DIR = Path("postprocessed")
OUT_DIR = POSTPROC_DIR / "contexts"

# окно дополнительных предложений вокруг упоминания
LEFT_WIN  = 1
RIGHT_WIN = 1

def load_json(p: Path) -> Any:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj: Any, p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def build_sentence_offsets(text: str, sentences: List[str]) -> List[Tuple[int,int]]:
    """Если offsets не сохранены в препроцессоре."""
    offs, pos = [], 0
    for s in sentences:
        idx = text.find(s, pos)
        if idx == -1:
            # fallback (редко): ищем по регулярке первые 10 символов
            m = re.search(re.escape(s[:10]), text[pos:])
            idx = pos + m.start() if m else pos
        offs.append((idx, idx+len(s)))
        pos = idx + len(s)
    return offs

def build_scene_cache(pre: dict) -> Dict[Tuple[str,str], dict]:
    cache = {}
    for ch, scenes in pre.items():
        for sc, data in scenes.items():
            text = data["cleaned_text"]
            sents = data.get("sentences") or []
            offs = data.get("sent_offsets")
            if not offs:
                offs = build_sentence_offsets(text, sents)
            cache[(str(ch), str(sc))] = {
                "text": text,
                "sentences": sents,
                "offsets": offs
            }
    return cache

def collect_context_for_char(char: dict,
                             scene_cache: Dict[Tuple[str,str], dict]) -> Dict[str, Any]:
    """
    Возвращает структуру с контекстами:
    {
      "id": ...,
      "norm": ...,
      "mentions": N,
      "events": [  # отсортированные по порядку
         {
           "chapter": "...",
           "scene": "...",
           "sent_id": 42,
           "sentence": "...",
           "window": ["prev sent", "sent", "next sent"],
           "text_span": [start, end]
         }, ...
      ],
      "scenes": {
         "(ch,sc)": "полный текст сцены" (опционально)
      }
    }
    """
    events = []
    # сортируем упоминания по (chapter, scene, sent_id, start)
    ments = sorted(char["mentions"], key=lambda m: (str(m["chapter"]), str(m["scene"]),
                                                    m.get("sent_id", -1), m.get("start", -1)))
    for m in ments:
        ch, sc = str(m["chapter"]), str(m["scene"])
        cache = scene_cache.get((ch, sc))
        if not cache:
            continue
        sents = cache["sentences"]
        offs  = cache["offsets"]
        sid   = m.get("sent_id")
        if sid is None or sid >= len(sents):
            # naive fallback: найдём предложение по start
            sid = 0
            start = m.get("start", 0)
            for i, (st, en) in enumerate(offs):
                if st <= start < en:
                    sid = i
                    break
        # окно вокруг
        left  = max(0, sid - LEFT_WIN)
        right = min(len(sents)-1, sid + RIGHT_WIN)
        window_sents = sents[left:right+1]

        events.append({
            "chapter": ch,
            "scene": sc,
            "sent_id": sid,
            "sentence": sents[sid],
            "window": window_sents,
            "text_span": [m.get("start"), m.get("end")],
            "mention_text": m.get("text"),
            "type": m.get("type", "name")
        })

    return {
        "id": char["id"],
        "norm": char["norm"],
        "gender": char.get("gender", "unknown"),
        "aliases": char.get("aliases", []),
        "mentions": len(char["mentions"]),
        "events": events
    }

def main(book_id: str = BOOK_ID):
    pre = load_json(PREPROC_DIR / f"{book_id}_preprocessed.json")
    # берём пропатченный файл персонажей
    chars = load_json(POSTPROC_DIR / f"{book_id}_characters_coref_patched.json")

    scene_cache = build_scene_cache(pre)

    contexts = {}
    for cid, ch in chars.items():
        contexts[cid] = collect_context_for_char(ch, scene_cache)

    out_path = OUT_DIR / f"{book_id}_contexts.json"
    save_json(contexts, out_path)
    logger.info(f"✓ Contexts saved → {out_path}")
    logger.info(f"Персонажей: {len(contexts)}")

if __name__ == "__main__":
    import sys
    bid = sys.argv[1] if len(sys.argv) > 1 else BOOK_ID
    main(bid)
