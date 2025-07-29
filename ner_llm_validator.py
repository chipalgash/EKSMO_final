# -*- coding: utf-8 -*-
"""
NER LLM Validator Stage: с помощью FRED‑T5 проверяем каждый кластер
и отсекаем те «сущности», которые не относятся к реальным персонажам.
Чтение:
  30_ner/<book_id>_ner.json
Запись:
  40_postprocess/<book_id>_ner_validated.json
"""
from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Dict, Any, List
from loguru import logger

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ------------- Классификатор на базе FRED‑T5 ----------------
class FredValidator:
    def __init__(self, model_name: str, device: str):
        logger.info(f"[ner_validator] Загружаю FRED‑T5 '{model_name}' на {device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model     = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        self.device    = device

    @torch.inference_mode()
    def classify(self, prompt: str, max_input: int = 512, max_gen: int = 64) -> str:
        inputs = self.tokenizer(
            prompt,
            truncation=True,
            max_length=max_input,
            return_tensors="pt"
        ).to(self.device)
        out_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_gen,
            num_beams=4,
            no_repeat_ngram_size=2
        )
        return self.tokenizer.decode(out_ids[0], skip_special_tokens=True)

# ------------- Утилиты для вычленения JSON из ответа -------------
def _extract_json(raw: str) -> dict:
    """
    Ищет первый {...} в тексте и пробует загрузить через json или json5.
    """
    m = re.search(r'(\{.*\})', raw, flags=re.DOTALL)
    if not m:
        return {}
    js = m.group(1)
    try:
        return json.loads(js)
    except json.JSONDecodeError:
        try:
            import json5  # если стоит, иначе нужно добавить зависимость
            return json5.loads(js)
        except Exception:
            return {}

# ------------- Шаблон промпта ----------------
PROMPT_TEMPLATE = """Ты — эксперт по персонажам художественных текстов.
Ниже даны фрагменты упоминаний одной сущности:

{context}

Определи, является ли это именно **персонажем** (человеком) в сюжете.
Ответь **строго** валидным JSON вида:
{{
  "is_character": true|false,
  "reason": "<короткое объяснение>"
}}
"""

# -------------------- STAGE --------------------
def run_stage(paths: Dict[str, Path], cfg: Dict[str, Any]) -> None:
    """
    Читает 30_ner/<book_id>_ner.json,
    проверяет каждую кластер-entity через LLM,
    сохраняет отфильтрованный результат в 40_postprocessed/<book_id>_ner_validated.json
    """
    book_id = paths["ner_dir"].name
    in_path = paths["ner_dir"] / f"{book_id}_ner.json"
    out_dir = paths.get("postproc_dir", paths["ner_dir"].parent / "40_postprocessed")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{book_id}_ner_validated.json"

    if not in_path.exists():
        logger.error(f"[ner_validator] Не найден входной файл: {in_path}")
        return

    # Загружаем кластеры из ner.json
    raw = json.loads(in_path.read_text(encoding="utf-8"))
    clusters: List[Dict[str, Any]] = raw.get("entities", [])

    # Параметры модели из config
    model_name = cfg.get("model_name", "ai-forever/FRED-T5-base")
    device     = cfg.get("device", "cpu")
    validator  = FredValidator(model_name, device)

    kept: List[Dict[str, Any]] = []
    for c in clusters:
        # Собираем контекст из первых 5 упоминаний
        snippets = [m["text"] for m in c.get("mentions", [])[:5]]
        context  = "\n".join(f"- {s}" for s in snippets)
        prompt   = PROMPT_TEMPLATE.format(context=context)
        raw_out  = validator.classify(prompt)
        js       = _extract_json(raw_out)
        is_char  = js.get("is_character", False)

        if is_char:
            kept.append(c)
        else:
            reason = js.get("reason", "")
            logger.info(f"[ner_validator] Отсекаем «{c.get('name')}»: {reason}")

    # Сохраняем только отфильтрованные кластеры
    result = {"book_id": book_id, "entities": kept}
    out_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    logger.info(f"[ner_validator] Осталось персонажей: {len(kept)}. Saved → {out_path.name}")
