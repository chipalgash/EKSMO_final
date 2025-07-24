# text_preprocessor.py
from __future__ import annotations
import re
import json
from pathlib import Path
from typing import Dict, Any, List
from loguru import logger

# быстрая русская сегментация предложений
try:
    from razdel import sentenize
except ImportError:
    sentenize = None

# лемматизация (опционально)
try:
    from pymorphy3 import MorphAnalyzer
    morph = MorphAnalyzer()
except Exception:
    morph = None


# ----------------- API для run_book ----------------- #
def run_stage(paths: Dict[str, Path], cfg: Dict[str, Any]):
    """
    paths: словарь путей (см. PathManager.as_dict())
    cfg:
        force: bool – пересчитать, даже если файл есть
        normalize: bool – делать лемматизацию
        min_sent_len: int – отсекать «предложения» короче n символов
    """
    book_id = paths["book_id"]
    in_dir = paths["extracted"]
    out_dir = paths["preprocessed"]
    out_dir.mkdir(parents=True, exist_ok=True)

    force = cfg.get("force", False)
    do_norm = cfg.get("normalize", False)
    min_len = cfg.get("min_sent_len", 3)

    out_path = out_dir / f"{book_id}_preprocessed.json"
    if out_path.exists() and not force:
        logger.info(f"[preprocess] {out_path.name} уже существует, пропускаю.")
        return

    # исходники
    txt_path = in_dir / f"{book_id}.txt"
    struct_path = in_dir / f"{book_id}_structured.json"

    if struct_path.exists():
        logger.info("[preprocess] Использую структурированный ввод (главы/сцены).")
        structured = json.loads(struct_path.read_text("utf-8"))
        result = preprocess_structured(structured, do_norm, min_len)
    elif txt_path.exists():
        logger.info("[preprocess] Использую плоский txt.")
        text = txt_path.read_text("utf-8")
        result = preprocess_flat(text, do_norm, min_len)
    else:
        logger.error(f"[preprocess] Нет входных файлов в {in_dir}")
        return

    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"[preprocess] Сохранено → {out_path}")


# ----------------- Основная логика ----------------- #
def preprocess_structured(structured: List[Dict[str, Any]], do_norm: bool, min_len: int) -> Dict[str, Any]:
    """
    structured: список глав [{title, scenes:[{id, text}]}]
    Возвращает единый JSON:
    {
      "book_id": ...,
      "chapters": [
        {"id": 1, "title": "...", "scenes":[
           {"id": 1, "text": "...", "sentences":[{"id":0,"text":"...","norm":"..."}]}
        ]}
      ],
      "flat_sentences": [...]
    }
    """
    chapters_out = []
    flat_sentences = []
    sent_global_id = 0

    for ch_id, ch in enumerate(structured, start=1):
        ch_title = ch.get("title", f"Глава {ch_id}")
        scenes_out = []
        for sc in ch.get("scenes", []):
            sc_id = sc.get("id", len(scenes_out) + 1)
            scene_text = sc.get("text", "")
            scene_text = clean_text(scene_text)
            sent_list = split_sentences(scene_text, min_len)

            sent_objs = []
            for idx, s in enumerate(sent_list):
                norm = normalize_sentence(s) if do_norm else ""
                sent_obj = {"id": idx, "text": s}
                if do_norm:
                    sent_obj["norm"] = norm
                sent_objs.append(sent_obj)

                flat_sentences.append({
                    "global_id": sent_global_id,
                    "chapter": ch_id,
                    "scene": sc_id,
                    "text": s,
                    **({"norm": norm} if do_norm else {})
                })
                sent_global_id += 1

            scenes_out.append({"id": sc_id, "text": scene_text, "sentences": sent_objs})

        chapters_out.append({"id": ch_id, "title": ch_title, "scenes": scenes_out})

    return {
        "chapters": chapters_out,
        "flat_sentences": flat_sentences
    }


def preprocess_flat(text: str, do_norm: bool, min_len: int) -> Dict[str, Any]:
    """
    Для случая, когда нет структуры глав/сцен.
    """
    text = clean_text(text)
    sent_list = split_sentences(text, min_len)

    flat_sentences = []
    for i, s in enumerate(sent_list):
        entry = {"global_id": i, "text": s}
        if do_norm:
            entry["norm"] = normalize_sentence(s)
        flat_sentences.append(entry)

    # всё в одной «главе» и одной «сцене»
    return {
        "chapters": [{
            "id": 1,
            "title": "Текст",
            "scenes": [{
                "id": 1,
                "text": text,
                "sentences": [
                    {"id": i, "text": s, **({"norm": flat_sentences[i]['norm']} if do_norm else {})}
                    for i, s in enumerate(sent_list)
                ]
            }]
        }],
        "flat_sentences": flat_sentences
    }


# ----------------- Вспомогательные функции ----------------- #
CLEAN_REPS = [
    (r"\s+", " "),               # многократные пробелы
    (r"[\u200b\u200e\uFEFF]", ""),  # скрытые юникод-символы
]

def clean_text(text: str) -> str:
    text = text.replace("\xa0", " ").replace("\t", " ")
    for pat, repl in CLEAN_REPS:
        text = re.sub(pat, repl, text)
    return text.strip()


def split_sentences(text: str, min_len: int) -> List[str]:
    """
    Разбивает на предложения. Если razdel не установлен — fallback простой точечной эвристикой.
    """
    sents: List[str] = []
    if sentenize:
        for s in sentenize(text):
            t = s.text.strip()
            if len(t) >= min_len:
                sents.append(t)
    else:
        # грубая эвристика
        parts = re.split(r"(?<=[.!?…])\s+", text)
        for p in parts:
            p = p.strip()
            if len(p) >= min_len:
                sents.append(p)
    return sents


WORD_RE = re.compile(r"[А-Яа-яA-Za-zЁё']+")

def normalize_sentence(sent: str) -> str:
    """
    Очень простая лемматизация: токенизируем слова, лемматизируем pymorphy3,
    собираем обратно строку. Если morph нет — возвращаем исходное предложение.
    """
    if morph is None:
        return sent
    tokens = WORD_RE.findall(sent)
    lemmas = []
    for tok in tokens:
        p = morph.parse(tok)
        if p:
            lemmas.append(p[0].normal_form)
        else:
            lemmas.append(tok.lower())
    return " ".join(lemmas)
