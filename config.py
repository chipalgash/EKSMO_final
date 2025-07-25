# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict

STAGE_ORDER = [
    "reader",       # docx -> txt (10_extracted)
    "preprocess",   # чистка/главы/сцены/предложения (20_preprocessed)
    "ner",          # Natasha+spaCy + постобработка (30_ner)
    "coref",        # правила coref (50_coref)
    "relations",    # граф отношений (60_relations)
    "contexts",     # сбор контекстов (70_contexts)
    "summary",      # FRED-T5/OpenAI (80_summaries)
]

# Модули лежат в корне проекта
STAGE_MODULES = {
    "reader":     "doc_reader",
    "preprocess": "text_preprocessor",
    "ner":        "ner_extractor",
    "coref":      "coref_resolver",
    "relations":  "relationships_extractor",
    "contexts":   "character_context_builder",
    "summary":    "character_summarizer",
}

STAGE_CFG: Dict[str, Dict[str, Any]] = {
    "global": {
        "workspace_root": "workspace",  # твой корень с книгами
        "log_level": "INFO",
    },
    "reader": {},
    "preprocess": {
        "sentencizer": "regex",
        "min_scene_len": 2,
        "strip_headers": True,
    },
    "ner": {
        "merge_spacy_natasha": True,
        "min_mentions": 3,
        "min_scenes": 2,
        "stopwords_person_like": ["глава","пролог","эпилог","часть","том","курсив","вечер","утро"],
        "fuzzy_threshold": 90,
        "use_morph_gender": True,
        "save_mentions_index": True,
    },
    "coref": {
        "window": 3,
        "pronouns": {
            "male":  ["он","его","нему","ним","него","им"],
            "female":["она","её","нее","ней","ею"],
            "neutral":["они","их","ими","них","им"],
        },
        "attach_type_field": True,
    },
    "relations": {
        "scene_min_cooccurs": 2,
        "sent_min_cooccurs": 1,
        "regex_roles": True,
    },
    "contexts": {
        "left_sentences": 2,
        "right_sentences": 1,
        "max_contexts_per_char": 100,
        "max_chars_per_context": 2500,
    },
    "summary": {
        "model": "fred_t5",          # fred_t5 | openai
        "chunk_tokens": 900,
        "cache_dir": "cache/llm",
        "prompt_template": "default",
        "save_book_summary": True,
    },
}


def deep_update(base: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in incoming.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_external_cfg(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".yml", ".yaml"}:
        import yaml
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    elif path.suffix.lower() == ".json":
        import json
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        raise ValueError(f"Unsupported config extension: {path.suffix}")
    if not isinstance(data, dict):
        raise ValueError("External config must be a dict at top-level")
    return data
