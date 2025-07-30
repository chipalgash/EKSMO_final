# Алгоритм сегмантации глав, выделения персонажей и суммаризации сюжетных линий

Комплексный пайплайн для автоматической обработки и суммаризации художественных текстов на русском языке.  
Разработано для магистерской работы по дисциплине «Фундаментальная и прикладная лингвистика» Томского государственного университета совместно с издательской компанией ЭКСМО (Пупкова М. А., Цымбалов Д. А.).

---

## 📦 Структура проекта

```
.
├── config.py                 # Центральная конфигурация пайплайна
├── run_book.py               # Оркестратор: запускает стадии по порядку
├── doc_reader.py             # 10_extracted — из .docx → txt + структура глав/сцен
├── text_preprocessor.py      # 20_preprocessed — чистка, сегментация на главы/сцены/предложения
├── ner_extractor.py          # 30_ner — NER (Natasha + spaCy) + пост-обработка, фильтры, кластеризация
├── ner_check.py              # 30_ner — проверка целостности NER-результатов
├── ner_llm_validator.py      # 40_postprocessed — LLM-валидация (FRED-T5) кластеров сущностей
├── coref_resolver.py         # 50_coref — привязка местоимений к именам (правила + neuralcoref)
├── relationships_extractor.py# 60_relations — построение графа отношений персонажей
├── character_context_builder.py  # 70_contexts — сбор контекстов упоминаний
└── character_summarizer.py   # 80_summaries — абстрактивная суммаризация (FRED-T5)
```

---

## 🚀 Быстрый старт

### 1. Клонировать репозиторий

```bash
git clone git@github.com:chipalgash/EKSMO_final.git
cd EKSMO_final
```

### 2. Установить зависимости

```bash
pip install -r requirements.txt
```

### 3. Положить файл книги

В каталог `workspace/<BOOK_ID>/10_extracted/` поместить `BOOK_ID.docx`.

### 4. Запустить весь пайплайн

```bash
python run_book.py --book-id 633450
```

### 5. Результаты

- Текст и структура: `workspace/633450/10_extracted/`
- Препроцессинг: `workspace/633450/20_preprocessed/`
- NER + индекс: `workspace/633450/30_ner/`
- NER-отчёт: `workspace/633450/30_ner/633450_ner_report.json`
- LLM-валидация: `workspace/633450/40_postprocessed/633450_ner_validated.json`
- Coreference: `workspace/633450/50_coref/`
- Отношения: `workspace/633450/60_relations/`
- Контексты: `workspace/633450/70_contexts/`
- Суммаризации: `workspace/633450/80_summaries/`

---

## 🔧 Подробное описание скриптов

### 1. doc_reader.py

- Преобразует `.docx` → чистый текст + JSON со структурой глав/сцен.
- Использует python-docx и регулярки для поиска заголовков.

Пример:

```bash
python doc_reader.py --book-id 633450
```

### 2. text_preprocessor.py

- Чистка служебных символов, унификация кавычек, токенизация.
- Разбивает текст на главы, сцены и предложения.

Ключевые параметры:
- `sentencizer`: regex — метод деления на предложения
- `min_scene_len`: минимальный размер сцены

### 3. ner_extractor.py

- Собирает все именованные сущности (модели Natasha + spaCy).
- Фильтрация «шума» (стоп-слова, слишком короткие/не-именованные фрагменты).
- Кластеризация по алиасам (RapidFuzz).
- Генерация уникальных `entity_id` и `mention_id` + сохранение индекса.

Выход:

```json
{
  "book_id": "633450",
  "entities": [ … ]
}
```

Индекс:

```json
{
  "1::2::5": ["ent_0:m_0", "ent_3:m_17", …]
}
```

### 4. ner_check.py

- Сравнивает NER-упоминания с `mentions_index.json`.
- Отчёт по сущностям и упоминаниям (`workspace/30_ner/<BOOK_ID>_ner_report.json`).

### 5. ner_llm_validator.py

- Валидирует кластеры с помощью модели FRED-T5.

Конфиг:

```yaml
ner_validator:
  sample_mentions: 3
  model_name: ai-forever/FRED-T5-base
  device: cuda
```

### 6. coref_resolver.py

- Связывает местоимения с ближайшими именами.
- Возможность включения neuralcoref.

### 7. relationships_extractor.py

- Строит граф персонажей по совместным упоминаниям и выделяет ролевые паттерны.

### 8. character_context_builder.py

- Собирает контексты предложений вокруг упоминаний.
- Результат — `contexts.json`.

### 9. character_summarizer.py

- Выполняет абстрактивную суммаризацию контекстов (FRED-T5).

---

## ⚙️ Конфигурация

В `config.py` можно менять:
- Порядок стадий (`STAGE_ORDER`)
- Пути (`DATA_DIR`, `SUBDIRS`)
- Параметры стадий (`STAGE_CFG`)

Пример:

```python
STAGE_CFG["summary"]["device"] = "cpu"
STAGE_CFG["ner"]["min_mentions"] = 5
```

---

## ✍️ Примеры использования

Запустить только NER и далее:

```bash
python run_book.py --book-id 633450 --start-from ner
```

Пропустить валидацию LLM:

```bash
python run_book.py --book-id 633450 --skip ner_validator
```

Повторить одну стадию:

```bash
python run_book.py --book-id 633450 --stop-after coref --force
```

---

## 📜 Итог

Этот пайплайн реализует полный цикл автоматизированного обзора художественного текста:

1. Экстракция
2. Препроцессинг
3. NER
4. Проверка и LLM-валидация
5. Coref
6. Отношения
7. Контексты
8. Абстрактивная суммаризация

---

Дипломный проект выполнен в рамках подготовки магистерской ВКР по дисциплине «Фундаментальная и прикладная лингвистика» на базе Томского государственного университета в сотрудничестве с издательской компанией ЭКСМО (Пупкова М. А., Цымбалов Д. А.).

