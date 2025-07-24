# doc_reader.py
from __future__ import annotations
import re
from pathlib import Path
from typing import List, Dict, Any
from loguru import logger
from docx import Document

# ----------------------  для run_book ---------------------- #
def run_stage(paths: Dict[str, Path], cfg: Dict[str, Any]):
    """
    Оркестратор вызывает эту функцию.
    paths: словарь путей из PathManager.as_dict()
    cfg:   параметры стадии (можно оставить пустым)
    """
    book_id = paths["book_id"]
    out_dir = paths["extracted"]
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_dir = paths["raw"]

    # ищем любой .docx в raw/
    docx_files = list(raw_dir.glob("*.docx"))
    if not docx_files:
        logger.error(f"[doc_reader] В {raw_dir} не найдено .docx. Укажи --docx или помести файл вручную.")
        return

    docx_path = docx_files[0]
    logger.info(f"[doc_reader] Используем файл: {docx_path.name}")

    force = cfg.get("force", False)
    save_structured = cfg.get("save_structured", True)
    do_segment = cfg.get("segment", True)

    flat_out = out_dir / f"{book_id}.txt"
    struct_out = out_dir / f"{book_id}_structured.json"

    if flat_out.exists() and (not force):
        logger.info(f"[doc_reader] {flat_out.name} уже есть, пропускаю извлечение.")
        return

    # 1. читаем docx
    text_blocks = read_docx_paragraphs(docx_path)

    # 2. соединяем в плоский текст
    full_text = "\n".join(text_blocks)

    # 3. (опционально) сегментируем в главы/сцены
    structured = None
    if do_segment:
        structured = segment_text(text_blocks)
        logger.info(f"[doc_reader] Найдено глав: {len(structured)}")

    # 4. сохраняем
    flat_out.write_text(full_text, encoding="utf-8")
    logger.info(f"[doc_reader] Текст сохранён → {flat_out}")

    if save_structured and structured is not None:
        import json
        with struct_out.open("w", encoding="utf-8") as f:
            json.dump(structured, f, ensure_ascii=False, indent=2)
        logger.info(f"[doc_reader] Структура (главы/сцены) сохранена → {struct_out}")


# ---------------------- внутренняя логика ---------------------- #
def read_docx_paragraphs(path: Path) -> List[str]:
    """
    Читает документ .docx и возвращает список непустых абзацев.
    """
    doc = Document(path)
    out = []
    for p in doc.paragraphs:
        t = p.text.strip()
        if t:
            out.append(t)
    return out


def segment_text(paragraphs: List[str]) -> List[Dict[str, Any]]:
    """
    Cегментация на главы/сцены.
    Возвращает список глав:
    [
      {
        "title": "Глава 1 ...",
        "scenes": [
           {"id": 1, "text": "..."},
           ...
        ]
      },
      ...
    ]

    Алгоритм простой:
    - Глава: строка, начинающаяся на "Глава", "Часть", римское число и т.п.
    - Сцена: пустая строка или маркер "***", "---" и т.п. (здесь — очень грубо).
    Доразовьём при необходимости.
    """
    chapters: List[Dict[str, Any]] = []
    current_chapter = None
    current_scene_lines: List[str] = []
    scene_id = 1

    chapter_pattern = re.compile(r"^\s*(ГЛАВА|Глава|Часть|ЧАСТЬ|[IVXLCM]{1,6}\.?)(\s|$)")
    scene_split_pattern = re.compile(r"^\s*(\*{3,}|-{3,}|#\s{0,})\s*$")

    def flush_scene():
        nonlocal current_scene_lines, scene_id, current_chapter
        if current_scene_lines:
            scene_text = "\n".join(current_scene_lines).strip()
            current_chapter["scenes"].append(
                {"id": scene_id, "text": scene_text}
            )
            scene_id += 1
            current_scene_lines = []

    for line in paragraphs:
        # новая глава?
        if chapter_pattern.match(line):
            # сбрасываем предыдущую главу
            if current_chapter:
                flush_scene()
                chapters.append(current_chapter)
            current_chapter = {"title": line.strip(), "scenes": []}
            scene_id = 1
            current_scene_lines = []
            continue

        # если нет активной главы - создаём "пролог"
        if current_chapter is None:
            current_chapter = {"title": "Пролог", "scenes": []}
            scene_id = 1

        # новая сцена?
        if scene_split_pattern.match(line):
            flush_scene()
            continue

        current_scene_lines.append(line)

    # финальный flush
    if current_chapter:
        flush_scene()
        chapters.append(current_chapter)

    return chapters
