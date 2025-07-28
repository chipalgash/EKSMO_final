# -*- coding: utf-8 -*-
from __future__ import annotations
import json
import json5
from pathlib import Path
from typing import Dict, Any, List
from loguru import logger

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# -------------------- PROMPTS --------------------
SYS_INSTR = (
    "Ты — литературный редактор. "
    "Твоя задача — кратко и структурированно описать персонажа. "
    "Игнорируй объекты, организации и предметы — интересуют только люди/персонажи. "
    "Пиши по-русски, научно-нейтральным стилем (без воды и клише)."
)

PROMPT_TEMPLATE = """{sys}

Фрагменты о персонаже «{name}» (хронологически):
{context}

Задачи:
1) Краткая биография персонажа (3–4 предложения).
2) Ключевые характеристики/черты личности (5–7 пунктов).
3) Хронология ключевых событий персонажа (bullet list, 6–10 пунктов, прошедшее время).
4) Итоговое резюме сюжетной линии персонажа (5–7 предложений).

Ответ строго в JSON:
{{
  "biography": "...",
  "traits": ["...", "..."],
  "timeline": ["...", "..."],
  "story_summary": "..."
}}
"""

# -------------------- UTILITIES --------------------
def enc_len(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))

def make_sent_chunks_by_tokens(
    lines: List[str],
    tokenizer,
    max_tokens: int,
    overlap: int
) -> List[str]:
    chunks: List[str] = []
    start = 0
    n = len(lines)
    while start < n:
        buf: List[str] = []
        tok_count = 0
        i = start
        while i < n:
            ln = lines[i]
            ln_tokens = enc_len(tokenizer, ln) + 1
            if buf and tok_count + ln_tokens > max_tokens:
                break
            buf.append(ln)
            tok_count += ln_tokens
            i += 1
        chunks.append("\n".join(buf))
        start = max(i - overlap, start + 1)
    return chunks

def safe_parse_json(text: str) -> dict:
    """Извлекает JSON-блок из произвольного текста."""
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("No JSON braces found")
    candidate = text[start:end+1]
    try:
        return json.loads(candidate)
    except Exception:
        return json5.loads(candidate)

def normalize_list_str(lst: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in lst:
        s = x.strip(" •-–—*").strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out

def merge_piece(merged: dict, piece: dict):
    if not merged["biography"] and piece.get("biography"):
        merged["biography"] = piece["biography"]
    if piece.get("traits"):
        merged["traits"].extend(piece["traits"])
    if piece.get("timeline"):
        merged["timeline"].extend(piece["timeline"])
    if piece.get("story_summary"):
        if merged["story_summary"]:
            merged["story_summary"] += " " + piece["story_summary"]
        else:
            merged["story_summary"] = piece["story_summary"]

def post_clean(merged: dict):
    merged["traits"] = normalize_list_str(merged["traits"])
    merged["timeline"] = normalize_list_str(merged["timeline"])
    merged["biography"] = merged["biography"].strip()
    merged["story_summary"] = merged["story_summary"].strip()

# -------------------- MODEL --------------------
class FredSummarizer:
    def __init__(self, model_name: str, device: str):
        logger.info(f"Loading FRED‑T5 '{model_name}' on {device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model     = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        self.device    = device

    @torch.inference_mode()
    def generate(self, prompt: str, max_input_tokens: int, max_gen_tokens: int) -> str:
        inputs = self.tokenizer(
            prompt,
            truncation=True,
            max_length=max_input_tokens,
            return_tensors="pt"
        ).to(self.device)
        out_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_gen_tokens,
            num_beams=4,
            no_repeat_ngram_size=3
        )
        return self.tokenizer.decode(out_ids[0], skip_special_tokens=True)

# -------------------- STAGE --------------------
def run_stage(paths: Dict[str, Path], cfg: Dict[str, Any]) -> None:
    book_id  = paths["book_root"].name
    ctx_path = paths["contexts_dir"]  / f"{book_id}_contexts.json"
    rel_path = paths["relations_dir"] / f"{book_id}_relationships.json"
    out_dir  = paths["summary_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    summ_cfg   = cfg.get("summary", {})
    model_type = summ_cfg.get("model", "fred_t5")
    device     = summ_cfg.get("device", "cpu")
    max_input  = summ_cfg.get("chunk_tokens", 900)
    max_gen    = summ_cfg.get("gen_tokens", 512)
    overlap    = summ_cfg.get("sent_overlap", 2)
    max_events = summ_cfg.get("max_events", 120)
    save_book  = summ_cfg.get("save_book_summary", False)

    # инициализация модели
    if model_type == "fred_t5":
        model_name = summ_cfg.get("model_name", "ai-forever/FRED-T5-large")
        summarizer = FredSummarizer(model_name, device)
    else:
        raise ValueError(f"Unsupported summary model: {model_type}")

    # -- Чтение contexts.json и подготовка списка (id, entity) --
    if not ctx_path.exists():
        logger.error(f"[summary] Contexts not found: {ctx_path}")
        return
    raw_data = json.loads(ctx_path.read_text(encoding="utf-8"))

    # Собираем итемы: поддерживаем dict и list
    if isinstance(raw_data, dict):
        items = list(raw_data.items())
    elif isinstance(raw_data, list):
        items = []
        for idx, ent in enumerate(raw_data):
            cid = str(ent.get("id", ent.get("entity_id", idx)))
            items.append((cid, ent))
    else:
        logger.error(f"[summary] Unexpected contexts format: {type(raw_data)}")
        return

    # генерируем summary персонажей
    characters_out: Dict[str, Any] = {}
    for cid, ent in items:
        name   = ent.get("norm", "")
        events = ent.get("events", [])[:max_events]
        if not events:
            continue

        lines = [
            f"[Гл.{e['chapter']} Сц.{e['scene']} #{e['sent_id']}] {e['sentence']}"
            for e in events
        ]
        chunks = make_sent_chunks_by_tokens(lines, summarizer.tokenizer, max_input, overlap)

        merged = {"biography": "", "traits": [], "timeline": [], "story_summary": ""}
        for chunk in chunks:
            prompt = PROMPT_TEMPLATE.format(sys=SYS_INSTR, name=name, context=chunk)
            raw    = summarizer.generate(prompt, max_input, max_gen)
            try:
                piece = safe_parse_json(raw)
            except Exception as e:
                logger.warning(f"[{name}] JSON parse error: {e}")
                continue
            merge_piece(merged, piece)
        post_clean(merged)

        characters_out[cid] = {"name": name, **merged}

    # сохраняем результаты персонажей
    char_path = out_dir / f"{book_id}_characters_summary.json"
    char_path.write_text(
        json.dumps(characters_out, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    logger.info(f"[summary] Characters saved → {char_path.name}")

    # опционально: summary всей книги
    if save_book:
        combined    = "\n".join(f"{v['name']}: {v['biography']}" for v in characters_out.values())
        book_prompt = "Общая история книги через призму персонажей:\n\n" + combined
        book_raw    = summarizer.generate(book_prompt, max_input * 2, max_gen)
        try:
            book_j      = safe_parse_json(book_raw)
            book_summary = book_j.get("story_summary", book_j.get("biography", "")).strip()
        except Exception:
            book_summary = book_raw.strip()

        book_path = out_dir / f"{book_id}_book_summary.json"
        book_path.write_text(
            json.dumps({"book_summary": book_summary}, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        logger.info(f"[summary] Book summary saved → {book_path.name}")
