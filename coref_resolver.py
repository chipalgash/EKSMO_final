# -*- coding: utf-8 -*-
"""
Правила coreference: местоимения -> ближайший персонаж.
Вход:
  30_ner/<book>_ner.json
  20_preprocessed/<book>_preprocessed.json
Выход:
  50_coref/<book>_coref.json
  50_coref/<book>_mentions_index.json
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Tuple, DefaultDict
from collections import defaultdict
import json
import re

from loguru import logger


def run_stage(paths: Dict[str, Path], cfg: Dict[str, Any]) -> None:
    book_id = paths["book_root"].name

    ner_path = paths["ner_dir"] / f"{book_id}_ner.json"
    idx_path = paths["ner_dir"] / f"{book_id}_mentions_index.json"
    prep_path = paths["preprocess_dir"] / f"{book_id}_preprocessed.json"

    out_dir = paths["coref_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    coref_path = out_dir / f"{book_id}_coref.json"
    coref_idx_path = out_dir / f"{book_id}_mentions_index.json"

    if coref_path.exists() and coref_idx_path.exists() and not cfg.get("force", False):
        logger.info(f"[coref] Уже есть {coref_path.name}, пропускаю.")
        return

    # ---- загрузка входов ----
    if not ner_path.exists():
        raise FileNotFoundError(f"[coref] Нет NER файла: {ner_path}")
    if not prep_path.exists():
        raise FileNotFoundError(f"[coref] Нет препроцесс-файла: {prep_path}")

    with ner_path.open("r", encoding="utf-8") as f:
        ner_data = json.load(f)
    with idx_path.open("r", encoding="utf-8") as f:
        mentions_index = json.load(f)
    with prep_path.open("r", encoding="utf-8") as f:
        prep_data = json.load(f)

    # ---- параметры ----
    window = int(cfg.get("window", 3))
    attach_type = bool(cfg.get("attach_type_field", True))
    pronouns_cfg = cfg.get("pronouns", {})
    pronoun2gender = build_pronoun_map(pronouns_cfg)

    # ---- индексация исходных именованных упоминаний ----
    entities = ner_data["entities"]
    # глобальный max mention counter
    max_mid = find_max_mention_number(entities)
    mention_counter = max_mid + 1

    # (chap,scene,sent) -> list of (entity_id, mention_id)
    name_mentions_map: DefaultDict[str, List[Tuple[int, str]]] = defaultdict(list)
    for ent in entities:
        for m in ent["mentions"]:
            if m.get("type") != "name":
                continue
            key = f"{m['chapter']}::{m['scene']}::{m['sent_id']}"
            name_mentions_map[key].append((ent["id"], m["mention_id"]))

    # ---- найдём местоимения по препроцесс-тексту ----
    pronoun_mentions: List[Dict[str, Any]] = []

    for ch in prep_data["chapters"]:
        ch_id = ch["id"]
        for sc in ch["scenes"]:
            sc_id = sc["id"]
            for sent in sc["sentences"]:
                sent_id = sent["id"]
                text = sent["text"]
                for match in find_pronouns(text, pronoun2gender):
                    token, start, end, gender = match
                    # решаем coref
                    target_eid = resolve_pronoun(
                        chapter=ch_id,
                        scene=sc_id,
                        sent_id=sent_id,
                        gender=gender,
                        window=window,
                        name_mentions_map=name_mentions_map,
                        entities=entities
                    )
                    if target_eid is None:
                        continue  # можно сохранить как нерешённые, если нужно
                    m_id = make_mention_id(target_eid, mention_counter)
                    mention_counter += 1

                    pm = {
                        "mention_id": m_id,
                        "entity_id": target_eid,
                        "chapter": ch_id,
                        "scene": sc_id,
                        "sent_id": sent_id,
                        "start": start,
                        "end": end,
                        "text": token,
                        "type": "pronoun" if attach_type else "coref",
                        "source": "coref_rule"
                    }
                    pronoun_mentions.append(pm)

                    # обновим индекс
                    key = f"{ch_id}::{sc_id}::{sent_id}"
                    mentions_index.setdefault(key, []).append(m_id)

    # ---- добавляем новые упоминания к сущностям ----
    added = 0
    for pm in pronoun_mentions:
        eid = pm["entity_id"]
        entities[eid]["mentions"].append(pm)
        added += 1

    # сортируем упоминания внутри каждого персонажа
    for ent in entities:
        ent["mentions"] = sorted(ent["mentions"], key=lambda x: (x["chapter"], x["scene"], x["sent_id"], x.get("start", 0)))

    # ---- сохранение ----
    out_obj = {"book_id": book_id, "entities": entities}
    with coref_path.open("w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2)

    with coref_idx_path.open("w", encoding="utf-8") as f:
        json.dump(mentions_index, f, ensure_ascii=False, indent=2)

    logger.info(f"[coref] Добавлено местоимённых упоминаний: {added}")
    logger.info(f"[coref] Сохранено: {coref_path.name}, {coref_idx_path.name}")


# -------------------- Вспомогательные --------------------

def build_pronoun_map(pronouns_cfg: Dict[str, List[str]]) -> Dict[str, str]:
    """
    {'male': [...], 'female':[...], 'neutral':[...]} -> {pronoun: 'male'|'female'|'neutral'}
    """
    mp = {}
    for g, lst in pronouns_cfg.items():
        for p in lst:
            mp[p.lower()] = g
    return mp


def find_pronouns(text: str, pronoun2gender: Dict[str, str]) -> List[Tuple[str, int, int, str]]:
    """
    Возвращает список (token, start, end, gender)
    """
    # соберём regex для всех местоимений: \b(он|его|...)\b
    # кэшировать глобально не обязательно, текст небольшой
    if not pronoun2gender:
        return []
    pattern = r"\b(" + "|".join(map(re.escape, pronoun2gender.keys())) + r")\b"
    res = []
    for m in re.finditer(pattern, text, flags=re.IGNORECASE):
        token = m.group(0)
        gender = pronoun2gender.get(token.lower(), "neutral")
        res.append((token, m.start(), m.end(), gender))
    return res


def resolve_pronoun(chapter: int,
                    scene: int,
                    sent_id: int,
                    gender: str,
                    window: int,
                    name_mentions_map: Dict[str, List[Tuple[int, str]]],
                    entities: List[Dict[str, Any]]) -> int | None:
    """
    Ищем ближайшее именованное упоминание в прошлых предложениях (до window).
    Проверяем совпадение пола, если известно.
    """
    # формируем список прошлых ключей (от текущего sent_id-1 назад)
    candidates: List[Tuple[int, str]] = []  # (eid, mention_id)
    for delta in range(0, window + 1):
        sid = sent_id - delta
        if sid < 0:
            break
        key = f"{chapter}::{scene}::{sid}"
        if key in name_mentions_map:
            candidates.extend(name_mentions_map[key])

    if not candidates:
        return None

    # сначала по гендеру
    if gender != "neutral":
        gender_matched = []
        for eid, mid in candidates:
            ent_g = entities[eid].get("gender")
            if ent_g == gender:
                gender_matched.append((eid, mid))
        if gender_matched:
            return gender_matched[-1][0]  # последний по времени

    # иначе берём последний вообще
    return candidates[-1][0]


def make_mention_id(entity_id: int, global_counter: int) -> str:
    return f"ent_{entity_id}:m_{global_counter}"


def find_max_mention_number(entities: List[Dict[str, Any]]) -> int:
    """
    Извлекаем максимальный номер m_* из mention_id.
    """
    mx = -1
    for e in entities:
        for m in e["mentions"]:
            mid = m.get("mention_id")
            if not mid:
                continue
            try:
                tail = mid.split(":m_")[-1]
                num = int(tail)
                if num > mx:
                    mx = num
            except Exception:
                continue
    return mx
