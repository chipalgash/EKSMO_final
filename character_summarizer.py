from __future__ import annotations
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple
from loguru import logger
import re
import json5
from rapidfuzz import process as rf_process, fuzz

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# -------------------- ARGS --------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--book_id", default="633450")
    p.add_argument("--ctx_dir", default="postprocessed/contexts")
    p.add_argument("--rel_path", default="postprocessed/633450_relationships_final_names.json")
    p.add_argument("--out_dir", default="postprocessed")
    p.add_argument("--model_name", default="ai-forever/FRED-T5-large")
    p.add_argument("--max_events", type=int, default=120)
    p.add_argument("--max_input_tokens", type=int, default=900)   # вход для модели
    p.add_argument("--max_gen_tokens", type=int, default=512)     # выход
    p.add_argument("--sent_overlap", type=int, default=2)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# -------------------- PROMPTS --------------------
SYS_INSTR = ("Ты — литературный редактор. Твоя задача — кратко и структурированно описать персонажа. "
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
# -------------------- IO --------------------
def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj: Any, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

# -------------------- CHUNKING --------------------
def enc_len(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))

def make_sent_chunks_by_tokens(
    lines: List[str],
    tokenizer,
    max_tokens: int,
    overlap: int
) -> List[str]:
    """Чанкуем по предложениям, следя за лимитом токенов."""
    chunks = []
    start = 0
    n = len(lines)
    while start < n:
        buf: List[str] = []
        i = start
        token_count = 0
        while i < n:
            ln = lines[i]
            ln_tokens = enc_len(tokenizer, ln) + 1  # за перенос строки
            if token_count + ln_tokens > max_tokens and buf:
                break
            buf.append(ln)
            token_count += ln_tokens
            i += 1
        chunks.append("\n".join(buf))
        start = max(i - overlap, start + 1)
    return chunks

# -------------------- JSON PARSING --------------------
def safe_parse_json(text: str) -> dict:
    """
    Пробуем вытащить JSON из произвольной строки (FRED-T5 может добавить текст вокруг).
    Сначала ищем ближайший блок {...}, потом json.loads / json5.loads.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        # fallback: попытка найти кавычки и т.п.
        raise ValueError("No JSON braces found")
    candidate = text[start:end+1]
    try:
        return json.loads(candidate)
    except Exception:
        # попробуем более либеральный парсер
        return json5.loads(candidate)

def normalize_list_str(lst: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in lst:
        if not x:
            continue
        s = x.strip(" •-–—*").strip()
        if not s:
            continue
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out

# -------------------- MODEL --------------------
class FredSummarizer:
    def __init__(self, model_name: str, device: str):
        logger.info(f"Loading model {model_name} on {device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        self.device = device

    @torch.inference_mode()
    def generate(self, prompt: str, max_input_tokens: int, max_gen_tokens: int) -> str:
        # обрезаем вход по токенам, если надо
        ids = self.tokenizer(
            prompt,
            truncation=True,
            max_length=max_input_tokens,
            return_tensors="pt"
        ).to(self.device)
        out_ids = self.model.generate(
            **ids,
            max_new_tokens=max_gen_tokens,
            do_sample=False,
            num_beams=4,
            length_penalty=1.0,
            no_repeat_ngram_size=3
        )
        return self.tokenizer.decode(out_ids[0], skip_special_tokens=True)

# -------------------- MERGE --------------------
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
    # biography & story_summary — просто strip
    if merged["biography"]:
        merged["biography"] = merged["biography"].strip()
    if merged["story_summary"]:
        merged["story_summary"] = merged["story_summary"].strip()

# -------------------- MAIN --------------------
def main():
    args = parse_args()

    ctx_path = Path(args.ctx_dir) / f"{args.book_id}_contexts.json"
    contexts = load_json(ctx_path)

    summarizer = FredSummarizer(args.model_name, args.device)

    results: Dict[str, Any] = {}
    for cid, ctx in contexts.items():
        name = ctx["norm"]
        events = ctx["events"][:args.max_events]
        if not events:
            continue

        # строки-предложения
        lines = [
            f"[Гл.{e['chapter']} Сц.{e['scene']} #{e.get('sent_id', -1)}] {e['sentence']}"
            for e in events
        ]

        chunks = make_sent_chunks_by_tokens(
            lines,
            tokenizer=summarizer.tokenizer,
            max_tokens=args.max_input_tokens,
            overlap=args.sent_overlap
        )

        merged = {
            "biography": "",
            "traits": [],
            "timeline": [],
            "story_summary": ""
        }

        for i, chunk in enumerate(chunks, 1):
            prompt = PROMPT_TEMPLATE.format(sys=SYS_INSTR, name=name, context=chunk)
            try:
                raw = summarizer.generate(prompt, args.max_input_tokens, args.max_gen_tokens)
                piece = safe_parse_json(raw)
            except Exception as e:
                logger.warning(f"[{name}] chunk {i}: {e}")
                continue
            merge_piece(merged, piece)

        post_clean(merged)

        results[cid] = {
            "name": name,
            "gender": ctx.get("gender", "unknown"),
            "aliases": ctx.get("aliases", []),
            **merged
        }

    out_path = Path(args.out_dir) / f"{args.book_id}_characters_llm.json"
    save_json(results, out_path)
    logger.info(f"✓ Saved: {out_path}")

if __name__ == "__main__":
    main()