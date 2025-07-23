import re
from pathlib import Path
from natasha import Segmenter, Doc
from pymorphy2 import MorphAnalyzer
from loguru import logger

segmenter = Segmenter()
morph = MorphAnalyzer()


def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[\u00ad\-­]', '', text)  # soft hyphen + обычный дефис
    text = text.replace('«', '"').replace('»', '"')
    logger.info("Текст очищен от лишних символов.")
    return text.strip()


def tokenize_sentences(text: str) -> list:
    doc = Doc(text)
    doc.segment(segmenter)
    sentences = [sent.text for sent in doc.sents if sent.text]
    logger.info(f"Текст токенизирован на {len(sentences)} предложений.")
    return sentences


def normalize_text(text: str) -> str:
    tokens = re.findall(r'\w+', text.lower())
    normalized_tokens = [morph.parse(token)[0].normal_form for token in tokens]
    normalized_text = ' '.join(normalized_tokens)
    logger.info("Текст нормализован (лемматизирован).")
    return normalized_text


def preprocess_text(text: str) -> dict:
    cleaned = clean_text(text)
    sentences = tokenize_sentences(cleaned)
    normalized = normalize_text(cleaned)

    return {
        "cleaned_text": cleaned,
        "sentences": sentences,
        "normalized_text": normalized
    }


def process_segmented_file(filepath: Path, output_dir: Path):
    with filepath.open('r', encoding='utf-8') as file:
        content = file.read()

    segments = re.split(r"(={5,} Глава .+? ={5,})", content)[1:]  # Сплит на главы
    structured_output = {}

    for i in range(0, len(segments), 2):
        chapter_title = segments[i].strip('=').strip()
        chapter_content = segments[i+1]

        scenes = re.split(r"(-{5,} Сцена .+? -{5,})", chapter_content)[1:]
        structured_output[chapter_title] = {}

        for j in range(0, len(scenes), 2):
            scene_title = scenes[j].strip('-').strip()
            scene_text = scenes[j+1].strip()

            preprocessed = preprocess_text(scene_text)

            structured_output[chapter_title][scene_title] = preprocessed

    save_preprocessed(structured_output, filepath.stem, output_dir)


def save_preprocessed(data: dict, filename: str, output_dir: Path):
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f"{filename}_preprocessed.json"
    import json
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    logger.info(f"Предобработанный текст сохранен в '{output_file}'")


if __name__ == "__main__":
    extracted_dir = Path("extracted_texts")
    preprocessed_dir = Path("preprocessed_texts")

    for filepath in extracted_dir.glob("*.txt"):
        logger.info(f"Обрабатывается файл '{filepath}'")
        process_segmented_file(filepath, preprocessed_dir)
