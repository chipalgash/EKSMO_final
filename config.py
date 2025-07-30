# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict

STAGE_ORDER = [
  "reader",
  "preprocess",
  "ner",
  "ner_check",
  "ner_validator",
  "coref",
  "relations",
  "contexts",
  "sort_contexts",
  "summary",
]


STAGE_MODULES = {
    "reader":        "doc_reader",
    "preprocess":    "text_preprocessor",
    "ner":           "ner_extractor",
    "ner_check":     "ner_check",
    "ner_validator": "ner_llm_validator",
    "coref":         "coref_resolver",
    "relations":     "relationships_extractor",
    "contexts": "character_context_builder",
    "sort_contexts": "context_sorter",
    "summary": "character_summarizer",
}

STAGE_CFG: Dict[str, Dict[str, Any]] = {
    "global": {
        "workspace_root": "workspace",
        "log_level":      "INFO",
    },
    "reader": {},
    "preprocess": {
        "sentencizer":    "regex",
        "min_scene_len":  2,
        "strip_headers":  True,
    },
    "ner": {
        "merge_spacy_natasha": True,
        "min_mentions":        3,
        "min_scenes":          2,
        "stopwords_person_like": [
            "глава","пролог","эпилог","часть",
            "том","курсив","вечер","утро"
        ],
        "fuzzy_threshold":     90,
        "use_morph_gender":    True,
        "save_mentions_index": True,
    },
    "ner_check": {
        "save_report": True,
    },
    "ner_validator": {
        "sample_mentions": 3,
        "model_name":      "ai-forever/FRED-T5-large",
        "device":          "cuda",
    },
    "coref": {
        "window": 3,
        "pronouns": {
            "male":   ["он","его","нему","ним","него","им"],
            "female": ["она","её","нее","ней","ею"],
            "neutral":["они","их","ими","них","им"],
        },
        "attach_type_field": True,
        "use_neural_coref":  True,
        "cross_scene":       False,
    },
    "relations": {
        "scene_min_cooccurs": 2,
        "sent_min_cooccurs":  1,
        "regex_roles":        True,
    },
    "contexts": {
        "left_sentences":        2,
        "right_sentences":       1,
        "max_contexts_per_char": 100,
        "max_chars_per_context": 2500,
    },
    "sort_contexts": {
        "enabled":   True,   # включить/отключить
        "top_chars": 5,      # сколько персонажей оставить (None = всех)
     },
    "summary": {
        "model":             "fred_t5",
        "model_name":        "ai-forever/FRED-T5-large",
        "device":            "cuda",
        "chunk_tokens":      900,
        "gen_tokens":        512,
        "sent_overlap":      2,
        "max_events":        120,
        "save_book_summary": True,
        "top_chars":         5,
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
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    elif path.suffix.lower() == ".json":
        import json
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        raise ValueError(f"Unsupported config extension: {path.suffix}")
    if not isinstance(data, dict):
        raise ValueError("External config must be a dict")
    return data
