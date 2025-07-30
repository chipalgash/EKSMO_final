# -*- coding: utf-8 -*-
from __future__ import annotations
import re
import json
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

Фрагменты о персонаже «{name}» (в хронологическом порядке):
{context}

Пожалуйста, составь ответ следующих блоков. Каждый блок обязательно отдели заголовком и пустой строкой:

Биография:
<3–4 предложения о прошлом персонажа>

Черты:
- trait1
- trait2
- trait3

Хронология:
- событие 1 (коротко)
- событие 2
- событие 3

Итоговый сюжет:
<5–7 предложений о роли и развитии персонажа в истории>

Никаких JSON, списков полей и лишнего текста — только эти четыре блока.
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

def normalize_list_str(lst: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in lst:
        s = x.strip(" •-–—*").strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out

def parse_summary_text(text: str) -> dict:
    pattern = (
        r"Биография:\s*(?P<bio>.*?)\n+"
        r"Черты:\s*(?P<traits>.*?)\n+"
        r"Хронология:\s*(?P<timeline>.*?)\n+"
        r"Итоговый сюжет:\s*(?P<story>.*)"
    )
    m = re.search(pattern, text, flags=re.DOTALL)
    if not m:
        return {
            "biography": "",
            "traits": [],
            "timeline": [],
            "story_summary": text.strip()
        }
    return {
        "biography":    m.group("bio").strip(),
        "traits":       re.findall(r"^[\-\*\u2022]\s*(.+)$", m.group("traits"), flags=re.M),
        "timeline":     re.findall(r"^[\-\*\u2022]\s*(.+)$", m.group("timeline"), flags=re.M),
        "story_summary": m.group("story").strip(),
    }

def merge_piece(merged: dict, piece: dict):
    if not merged["biography"] and piece.get("biography"):
        merged["biography"] = piece["biography"]
    merged["traits"].extend(piece.get("traits", []))
    merged["timeline"].extend(piece.get("timeline", []))
    if piece.get("story_summary"):
        merged["story_summary"] = (
            (merged["story_summary"] + " " + piece["story_summary"])
            if merged["story_summary"] else piece["story_summary"]
        )

def post_clean(merged: dict):
    merged["traits"]        = normalize_list_str(merged["traits"])
    merged["timeline"]      = normalize_list_str(merged["timeline"])
    merged["biography"]     = merged["biography"].strip()
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
    book_id   = paths["book_root"].name
    ctx_path  = paths["contexts_dir"]  / f"{book_id}_contexts.json"
    out_dir   = paths["summary_dir"]; out_dir.mkdir(parents=True, exist_ok=True)

    summ_cfg   = cfg.get("summary", {})
    model_type = summ_cfg.get("model", "fred_t5")
    device     = summ_cfg.get("device", "cuda")
    max_input  = summ_cfg.get("chunk_tokens", 900)
    max_gen    = summ_cfg.get("gen_tokens", 512)
    overlap    = summ_cfg.get("sent_overlap", 2)
    max_events = summ_cfg.get("max_events", 120)
    save_book  = summ_cfg.get("save_book_summary", False)
    top_n      = summ_cfg.get("top_chars", None)

    if model_type != "fred_t5":
        raise ValueError(f"Unsupported summary model: {model_type}")
    model_name = summ_cfg.get("model_name", "ai-forever/FRED-T5-large")
    summarizer = FredSummarizer(model_name, device)

    if not ctx_path.exists():
        logger.error(f"[summary] Contexts not found: {ctx_path}")
        return
    raw_data = json.loads(ctx_path.read_text(encoding="utf-8"))

    # распаковываем payload
    if isinstance(raw_data, dict) and "contexts" in raw_data:
        ctx_list = raw_data["contexts"]
    elif isinstance(raw_data, list):
        ctx_list = raw_data
    else:
        logger.error(f"[summary] Неподдерживаемый формат {ctx_path}")
        return

    logger.info(f"[summary] Из контекстов получили {len(ctx_list)} кандидатов")

    # топ‑N по числу событий
    if isinstance(top_n, int) and top_n > 0:
        ctx_list = sorted(
            ctx_list,
            key=lambda ent: len(ent.get("events", [])),
            reverse=True
        )[:top_n]
        logger.info(f"[summary] Оставляем топ‑{top_n} по events: {len(ctx_list)} персонажей")

    # теперь формируем items для суммаризации
    items = [
        (str(ent.get("entity_id", ent.get("id", ""))), ent)
        for ent in ctx_list
    ]
    total_chars = len(items)
    logger.info(f"[summary] Начинаем суммаризацию {total_chars} персонажей")

    characters_out: Dict[str, Any] = {}
    for idx, (cid, ent) in enumerate(items, start=1):
        name   = ent.get("name") or ent.get("norm") or ""
        logger.info(f"[summary] Персонаж {idx}/{total_chars} — id={cid}, name={name}")

        events = ent.get("events", [])[:max_events]
        if not events:
            logger.warning(f"[{cid}] Нет событий, пропускаю")
            continue

        # собираем строки и разбиваем на чанки
        lines = [
            f"[Гл.{e['chapter']} Сц.{e['scene']} #{e['sent_id']}] {e['text']}"
            for e in events
        ]
        chunks = make_sent_chunks_by_tokens(lines, summarizer.tokenizer, max_input, overlap)
        logger.info(f"[{cid}] Разбито на {len(chunks)} чанков")

        merged = {"biography": "", "traits": [], "timeline": [], "story_summary": ""}
        for j, chunk in enumerate(chunks, start=1):
            logger.info(f"[{cid}] Обрабатываю чанк {j}/{len(chunks)}")
            prompt = PROMPT_TEMPLATE.format(sys=SYS_INSTR, name=name, context=chunk)
            raw    = summarizer.generate(prompt, max_input, max_gen)
            piece  = parse_summary_text(raw)
            merge_piece(merged, piece)
            logger.info(f"[{cid}] Чанк {j}/{len(chunks)} обработан")

        post_clean(merged)
        logger.info(f"[{cid}] Готов итоговый summary персонажа")
        characters_out[cid] = {"name": name, **merged}

    # сохраняем summaries
    char_path = out_dir / f"{book_id}_characters_summary.json"
    char_path.write_text(json.dumps(characters_out, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"[summary] Characters saved → {char_path.name}")

    # опционально общий summary книги
    if save_book:
        combined   = "\n".join(f"{v['name']}: {v['biography']}" for v in characters_out.values())
        book_prompt = "Общая история книги через призму персонажей:\n\n" + combined
        book_raw    = summarizer.generate(book_prompt, max_input*2, max_gen)
        book_summary = book_raw.strip()
        book_path = out_dir / f"{book_id}_book_summary.json"
        book_path.write_text(json.dumps({"book_summary": book_summary}, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"[summary] Book summary saved → {book_path.name}")
