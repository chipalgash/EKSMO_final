# relationships_refine.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, Tuple
from loguru import logger

BOOK_ID = "633450"
POSTPROC_DIR = Path("postprocessed")

# Пороги
MIN_SCENE = 3
MIN_SENT  = 2

LABEL_PRIORITY = {"regex": 3, "cooc_sent": 2, "cooc_scene": 1}

def loadj(p: Path) -> Any:
    return json.loads(p.read_text("utf-8"))

def savej(obj: Any, p: Path):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), "utf-8")

def pick_label(cur_label: str, new_label: str) -> str:
    return new_label if LABEL_PRIORITY.get(new_label, 0) > LABEL_PRIORITY.get(cur_label, 0) else cur_label

def short_name(full: str) -> str:
    # оставим последнее слово и, если есть, первое (Имя Фамилия)
    parts = full.split()
    if len(parts) >= 2:
        return parts[0] + " " + parts[-1]
    return full

def refine():
    rel_path  = POSTPROC_DIR / f"{BOOK_ID}_relationships_final.json"
    chars_path = POSTPROC_DIR / f"{BOOK_ID}_characters_coref_patched.json"
    out_ids    = POSTPROC_DIR / f"{BOOK_ID}_relationships_final_ids.json"
    out_names  = POSTPROC_DIR / f"{BOOK_ID}_relationships_final_names.json"

    rels = loadj(rel_path)          # {charA: {charB: {label, weight, evidence}}}
    chars = loadj(chars_path)

    # 1. Симметризация и фильтрация
    seen_pairs: Dict[Tuple[str,str], Dict[str, Any]] = {}

    for a, neigh in rels.items():
        for b, info in neigh.items():
            if a == b:
                continue
            # фильтр по весу для cooc_* (regex оставляем всегда)
            if info["label"].startswith("cooc"):
                thr = MIN_SENT if info["label"] == "cooc_sent" else MIN_SCENE
                if info["weight"] < thr:
                    continue

            pair = tuple(sorted([a, b]))
            cur = seen_pairs.get(pair)
            if cur is None:
                seen_pairs[pair] = {
                    "a": pair[0],
                    "b": pair[1],
                    "label": info["label"],
                    "weight": info["weight"],
                    "evidence": info.get("evidence", [])
                }
            else:
                # объединяем
                cur["label"] = pick_label(cur["label"], info["label"])
                cur["weight"] = max(cur["weight"], info["weight"])
                if info.get("evidence"):
                    cur["evidence"].extend(info["evidence"])

    # 2. Карта id → имя
    id2name = {cid: chars[cid]["norm"] for cid in chars}
    id2short = {cid: short_name(chars[cid]["norm"]) for cid in chars}

    # 3. Форматы
    # 3.1 ID-формат (симметричный список)
    edges_ids = list(seen_pairs.values())

    # 3.2 Имя-формат (словарь name -> {name: {...}})
    edges_by_name: Dict[str, Dict[str, Any]] = {}
    for e in edges_ids:
        a, b = e["a"], e["b"]
        name_a = id2short.get(a, a)
        name_b = id2short.get(b, b)
        edges_by_name.setdefault(name_a, {})[name_b] = {
            "label": e["label"],
            "weight": e["weight"],
            "evidence": e.get("evidence", [])
        }
        edges_by_name.setdefault(name_b, {})[name_a] = {
            "label": e["label"],
            "weight": e["weight"],
            "evidence": e.get("evidence", [])
        }

    # 4. Сохраняем
    savej(edges_ids, out_ids)
    savej(edges_by_name, out_names)
    logger.info(f"✓ Saved:\n  {out_ids}\n  {out_names} (human-readable)")

if __name__ == "__main__":
    refine()
