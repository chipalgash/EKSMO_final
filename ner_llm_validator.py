# -*- coding: utf-8 -*-
"""
NER LLM Validator Stage:
  Читает:  30_ner/<book_id>_ner.json
  Пишет:   40_postprocessed/<book_id>_ner_validated.json

Для каждой сущности берёт несколько сэмплов упоминаний и через FRED‑T5
спрашивает: «является ли это действующим персонажем художественного текста?»
Если ответ «Да», оставляем сущность, иначе — отбрасываем.
"""

from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Dict, Any, List
from loguru import logger

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# -------------------- LLM КЛАССИФИКАТОР --------------------
class FredClassifier:
    def __init__(self, model_name: str, device: str):
        logger.info(f"[ner_validator] Loading FRED‑T5 '{model_name}' on {device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model     = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        self.device    = device

    @torch.inference_mode()
    def classify(self, prompt: str, max_input: int, max_gen: int) -> str:
        inputs = self.tokenizer(
            prompt,
            truncation=True,
            max_length=max_input,
            return_tensors="pt"
        ).to(self.device)
        out = self.model.generate(
            **inputs,
            max_new_tokens=max_gen,
            num_beams=3,
            no_repeat_ngram_size=2
        )
        return self.tokenizer.decode(out[0], skip_special_tokens=True).strip()

# -------------------- STAGE --------------------
def run_stage(paths: Dict[str, Path], cfg: Dict[str, Any]) -> None:
    # пути
    book_id  = paths["ner_dir"].name
    ner_path = paths["ner_dir"] / f"{book_id}_ner.json"
    out_dir  = paths["postprocess_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{book_id}_ner_validated.json"

    # проверяем файл
    if not ner_path.exists():
        logger.error(f"[ner_validator] Нет входного NER: {ner_path}")
        return

    # конфиг для валидации
    val_cfg         = cfg.get("ner_validator", {})
    use_llm         = val_cfg.get("use_llm", True)
    sample_mentions = val_cfg.get("sample_mentions", 3)
    model_name      = val_cfg.get("model_name", "ai-forever/FRED-T5-base")
    device          = val_cfg.get("device", "cuda")
    max_input       = val_cfg.get("chunk_tokens", 256)
    max_gen         = val_cfg.get("gen_tokens", 8)

    # загружаем NER
    ner_data = json.loads(ner_path.read_text(encoding="utf-8"))
    entities = ner_data.get("entities", [])

    # инициализируем классификатор
    classifier = None
    if use_llm:
        classifier = FredClassifier(model_name, device)

    valid_entities: List[Dict[str, Any]] = []
    total = len(entities)
    logger.info(f"[ner_validator] Всего сущностей: {total}")

    for i, ent in enumerate(entities, start=1):
        ent_id = ent.get("id")
        norm   = ent.get("norm", "")
        mentions = ent.get("mentions", [])
        logger.info(f"[ner_validator] [{i}/{total}] Проверяю '{norm}' (id={ent_id})")

        # если LLM выключен — сразу считаем валидным
        if not use_llm:
            valid = True
        else:
            # выбираем несколько первых упоминаний
            samples = [m.get("text","") for m in mentions[:sample_mentions]]
            context = "\n".join(f"- {s}" for s in samples if s)
            prompt = (
                "У нас художественный текст. Ниже фрагменты с упоминаниями "
                f"имени «{norm}»:\n{context}\n\n"
                "Является ли эта сущность ПЕРСОНАЖЕМ (человек) "
                "в описываемом произведении? Ответь одним словом: Да или Нет."
            )
            raw = classifier.classify(prompt, max_input, max_gen)
            # ищем «да»
            if re.search(r"\bда\b", raw, flags=re.IGNORECASE):
                valid = True
            else:
                valid = False
            logger.info(f"[ner_validator] → LLM ответ: '{raw.splitlines()[0]}' => {'keep' if valid else 'drop'}")

        if valid:
            valid_entities.append(ent)

    # сохраняем отфильтрованные сущности
    out = {"entities": valid_entities}
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"[ner_validator] Сохранено валидных сущностей: {len(valid_entities)} из {total}")
