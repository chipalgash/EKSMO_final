import re
from pathlib import Path
from docx import Document
from loguru import logger

def read_docx(filepath: str) -> dict:
    """
    Читает .docx-файл и возвращает структурированный текст, разбитый на главы и сцены.

    Args:
        filepath (str): путь к файлу .docx

    Returns:
        dict: словарь с текстом, разбитым на главы и сцены
    """
    try:
        doc = Document(filepath)
        paragraphs = [para.text.strip() for para in doc.paragraphs if para.text.strip()]
        logger.info(f"Документ '{filepath}' успешно загружен.")

        structured_text = segment_chapters_and_scenes(paragraphs)
        save_structured_text(filepath, structured_text)

        return structured_text

    except Exception as e:
        logger.error(f"Ошибка при чтении документа '{filepath}': {e}")
        return {}

def segment_chapters_and_scenes(paragraphs: list) -> dict:
    """
    Сегментирует текст на главы и сцены с помощью регулярных выражений.

    Args:
        paragraphs (list): список абзацев из документа

    Returns:
        dict: структура вида {chapter_num: {scene_num: [paragraphs]}}
    """
    chapters = {}
    current_chapter = '0'
    current_scene = '0'

    chapter_pattern = re.compile(r'^\s*(глава|часть)\s+(\w+)', re.IGNORECASE)
    scene_pattern = re.compile(r'^(?:сцена|эпизод)\s+(\d+)|^\*\*\*|^\-\-\-$', re.IGNORECASE)

    for para in paragraphs:
        chapter_match = chapter_pattern.match(para.lower())
        scene_match = scene_pattern.match(para.lower())

        if chapter_match:
            current_chapter = chapter_match.group(2)
            current_scene = '0'
            chapters[current_chapter] = {}
            logger.debug(f"Найдена глава: {current_chapter}")

        elif scene_match:
            current_scene = str(int(current_scene) + 1)
            chapters[current_chapter][current_scene] = []
            logger.debug(f"Найдена сцена: {current_scene} (глава {current_chapter})")

        else:
            if current_chapter not in chapters:
                chapters[current_chapter] = {}
            if current_scene not in chapters[current_chapter]:
                chapters[current_chapter][current_scene] = []
            chapters[current_chapter][current_scene].append(para)

    logger.info(f"Документ сегментирован на {len(chapters)} глав.")
    return chapters

def save_structured_text(filepath: str, structured_text: dict, output_dir: str = "extracted_texts"):
    """
    Сохраняет структурированный текст в текстовый файл.

    Args:
        filepath (str): путь исходного файла .docx
        structured_text (dict): структурированный текст (главы и сцены)
        output_dir (str): папка для сохранения результатов
    """
    Path(output_dir).mkdir(exist_ok=True)

    filename = Path(filepath).stem
    output_file = Path(output_dir) / f"{filename}.txt"

    with output_file.open("w", encoding="utf-8") as f:
        for chapter, scenes in structured_text.items():
            f.write(f"\n{'='*20} Глава {chapter} {'='*20}\n\n")
            for scene, paragraphs in scenes.items():
                f.write(f"{'-'*10} Сцена {scene} {'-'*10}\n")
                for para in paragraphs:
                    f.write(para + "\n")
                f.write("\n")

    logger.info(f"Результаты сегментации сохранены в {output_file}")

if __name__ == "__main__":
    filepath = "/Users/pk/Desktop/датасет вкр извлеченный/633450.docx"  # путь к файлу
    structured_text = read_docx(filepath)
