# coref_resolver.py
from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple
from loguru import logger

PRONOUNS = {
    "он": "male", "его": "male", "ему": "male", "ним": "male", "него": "male",
    "она": "female", "её": "female", "ей": "female", "неё": "female", "ею": "female",
    "они": "plur", "их": "plur", "им": "plur", "ими": "plur", "них": "plur"
}
WINDOW = 3  # глубина поиска назад по предложениям

def load_json(p: Path) -> Any:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj: Any, p: Path):
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def build_scene_index(chars: Dict[str, dict]) -> Dict[Tuple[str,str], List[dict]]:
    """Сгруппировать все именованные mentions по сценам."""
    idx = {}
    for cid, ch in chars.items():
        for m in ch["mentions"]:
            m = dict(m)
            m["char_id"] = cid
            key = (m["chapter"], m["scene"])
            idx.setdefault(key, []).append(m)
    return idx

def coref_scene(scene_text: str,
                sentences: List[str],
                offsets: List[Tuple[int,int]],
                explicit_mentions: List[dict],
                genders: Dict[str, str]) -> List[dict]:
    """
    Возвращает новые pronoun-mentions.
    explicit_mentions: уже найденные именованные упоминания в сцене (с id/char_id/start/end/sent_id)
    """
    # по предложениям соберём, какие персонажи уже «видны»
    sent_to_chars = {i: [] for i in range(len(sentences))}
    for m in explicit_mentions:
        sid = m.get("sent_id")
        if sid is not None:
            sent_to_chars[sid].append(m["char_id"])

    new_mentions = []
    # Проходим предложения
    for i, sent in enumerate(sentences):
        tokens = re.finditer(r"\w+", sent.lower())
        for tok in tokens:
            t = tok.group(0)
            if t not in PRONOUNS:
                continue
            need_g = PRONOUNS[t]
            target = None

            # ищем в окне назад подходящего персонажа
            for back in range(0, WINDOW+1):
                sid = i - back
                if sid < 0:
                    break
                for cid in reversed(sent_to_chars.get(sid, [])):
                    g = genders.get(cid, "unknown")
                    if need_g == "plur" or g == need_g:
                        target = cid
                        break
                if target:
                    break

            if not target:
                continue

            # вычислим абсолютные offsets
            sent_abs_start = offsets[i][0]
            start = sent_abs_start + tok.start()
            end = sent_abs_start + tok.end()

            new_mentions.append({
                "chapter": explicit_mentions[0]["chapter"] if explicit_mentions else None,
                "scene": explicit_mentions[0]["scene"]     if explicit_mentions else None,
                "start": start,
                "end": end,
                "text": t,
                "type": "pronoun",
                "sent_id": i,
                "start_in_sent": tok.start(),
                "end_in_sent": tok.end(),
                "char_id": target
            })

            # добавим этот персонаж как «последний» для следующих местоимений
            sent_to_chars[i].append(target)

    return new_mentions

def enrich_with_coref(book_id: str,
                      preproc_dir: Path = Path("preprocessed_texts"),
                      postproc_dir: Path = Path("postprocessed")):
    pre = load_json(preproc_dir / f"{book_id}_preprocessed.json")
    chars = load_json(postproc_dir / f"{book_id}_characters_clean_enriched.json")

    # карта сцен -> именованные упоминания
    scene_index = build_scene_index(chars)

    # genders
    genders = {cid: ch.get("gender", "unknown") for cid, ch in chars.items()}

    added = []
    # для нумерации новых mentions
    counts = {cid: len(v["mentions"]) for cid, v in chars.items()}

    for chapter, scenes in pre.items():
        for scene, data in scenes.items():
            key = (chapter, scene)
            text = data["cleaned_text"]
            sentences = data.get("sentences") or []
            offsets = data.get("sent_offsets")  # если были сохранены
            if not sentences:
                # fallback
                sentences = re.split(r'(?<=[.!?…])\\s+', text)
                sentences = [s.strip() for s in sentences if s.strip()]
            if not offsets:
                # соберём как в enricher
                offs = []
                pos = 0
                for s in sentences:
                    idx = text.find(s, pos)
                    offs.append((idx, idx + len(s)))
                    pos = idx + len(s)
                offsets = offs

            explicit = scene_index.get(key, [])
            # безопасность: проставим chapter/scene в explicit, если вдруг нет
            for m in explicit:
                m.setdefault("chapter", chapter)
                m.setdefault("scene", scene)

            coref_new = coref_scene(text, sentences, offsets, explicit, genders)

            # записываем
            for m in coref_new:
                cid = m["char_id"]
                counts[cid] += 1
                m["id"] = f"{cid}_m{counts[cid]}"
                chars[cid]["mentions"].append(m)
            added.extend(coref_new)

    # сохраняем
    out_chars = postproc_dir / f"{book_id}_characters_coref.json"
    out_added = postproc_dir / f"{book_id}_coref_mentions.json"
    save_json(chars, out_chars)
    save_json(added, out_added)

    logger.info(f"✓ Coref added: {len(added)} mentions")
    logger.info(f"→ {out_chars}")
    logger.info(f"→ {out_added}")

if __name__ == "__main__":
    import sys
    bid = sys.argv[1] if len(sys.argv) > 1 else "633450"
    enrich_with_coref(bid)
