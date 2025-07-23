# relationships_postprocess.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, List
from loguru import logger

BOOK_ID = "633450"
POSTPROC_DIR = Path("postprocessed")

# Пороговые значения — подстрой под свой корпус
SCENE_MIN = 3   # минимум совместных сцен
SENT_MIN  = 2   # минимум совместных предложений

# Приоритет меток: если есть regex-роль, оставляем её; иначе cooc_*
ROLE_PRIORITY = {"regex": 2, "cooc_sent": 1, "cooc_scene": 0}

def load_json(p: Path) -> Any:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj: Any, p: Path):
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def main(book_id=BOOK_ID):
    raw_path = POSTPROC_DIR / f"{book_id}_relations_raw.json"
    chars_path = POSTPROC_DIR / f"{book_id}_characters_coref_patched.json"
    out_path = POSTPROC_DIR / f"{book_id}_relationships_final.json"

    rel_raw = load_json(raw_path)
    chars = load_json(chars_path)

    cooc_scene = rel_raw["cooc_scene"]
    cooc_sent = rel_raw["cooc_sent"]
    regex_edges = rel_raw["regex_relations"]

    # 1) Собираем кандидатов из cooc с порогами
    edges_map: Dict[str, Dict[str, Dict[str, Any]]] = {}
    # cooc_scene
    for a, neigh in cooc_scene.items():
        for b, w in neigh.items():
            if w >= SCENE_MIN:
                e = edges_map.setdefault(a, {}).setdefault(b, {"label": "cooc_scene", "weight": 0, "evidence": []})
                e["weight"] = max(e["weight"], w)

    # cooc_sent
    for a, neigh in cooc_sent.items():
        for b, w in neigh.items():
            if w >= SENT_MIN:
                e = edges_map.setdefault(a, {}).setdefault(b, {"label": "cooc_sent", "weight": 0, "evidence": []})
                # если уже была сцена, выбираем более сильную «cooc_sent»
                if ROLE_PRIORITY["cooc_sent"] > ROLE_PRIORITY.get(e["label"], -1):
                    e["label"] = "cooc_sent"
                e["weight"] = max(e["weight"], w)

    # regex (roles)
    for r in regex_edges:
        a, b, label = r["a"], r["b"], r["label"]
        e = edges_map.setdefault(a, {}).setdefault(b, {"label": label, "weight": 0, "evidence": []})
        if ROLE_PRIORITY["regex"] > ROLE_PRIORITY.get(e["label"], -1):
            e["label"] = label
        # добавляем evidence
        e["evidence"].append({
            "chapter": r["chapter"],
            "scene": r["scene"],
            "span": r["span"],
            "text": r["evidence_text"]
        })
        # для regex ставим вес = кол-во срабатываний regex
        e["weight"] += 1

    # 2) Превращаем в финальную структуру:
    # {
    #   "charA": {"charB": {"label": "...", "weight": n, "evidence": [...]}, ...},
    #   ...
    # }
    relationships: Dict[str, Dict[str, Any]] = {}
    for a, neigh in edges_map.items():
        for b, info in neigh.items():
            relationships.setdefault(a, {})[b] = info

    # 3) Сохраняем
    save_json(relationships, out_path)
    logger.info(f"✓ Final relationships saved → {out_path}")

if __name__ == "__main__":
    import sys
    bid = sys.argv[1] if len(sys.argv) > 1 else BOOK_ID
    main(bid)
