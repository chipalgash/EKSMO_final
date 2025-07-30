# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from config import (
    STAGE_ORDER,
    STAGE_MODULES,
    STAGE_CFG,
    deep_update,
    load_external_cfg,
)


class PathManager:
    """
    Layout:
      workspace/<book_id>/
        00_raw/
        10_extracted/          reader
        20_preprocessed/       preprocess
        30_ner/
        50_coref/
        60_relations/
        70_contexts/
        80_summaries/
        logs/
    """
    def __init__(self, workspace_root: Path, book_id: str):
        self.workspace_root = workspace_root
        self.book_id = book_id
        self.book_root = workspace_root / book_id

        self.stage_dirs = {
            "reader":          self.book_root / "10_extracted",
            "preprocess":      self.book_root / "20_preprocessed",
            "ner":             self.book_root / "30_ner",
            "coref":           self.book_root / "50_coref",
            "relations":       self.book_root / "60_relations",
            "contexts":        self.book_root / "70_contexts",
            "filter_contexts": self.book_root / "72_filtered_contexts",
            "summary":         self.book_root / "80_summaries",
        }

        self.raw_dir  = self.book_root / "00_raw"
        self.logs_dir = self.book_root / "logs"

        for p in [self.workspace_root, self.book_root, self.logs_dir]:
            p.mkdir(parents=True, exist_ok=True)

    def dir(self, stage: str) -> Path:
        d = self.stage_dirs[stage]
        d.mkdir(parents=True, exist_ok=True)
        return d

    def paths_dict(self) -> Dict[str, Path]:
        pd: Dict[str, Path] = {
            "workspace_root": self.workspace_root,
            "book_root": self.book_root,
            "raw_dir": self.raw_dir,
            "logs_dir": self.logs_dir,
        }
        # стандартные ключи <stage>_dir + короткие
        for stage, p in self.stage_dirs.items():
            pd[f"{stage}_dir"] = self.dir(stage)
            pd[stage] = pd[f"{stage}_dir"]

        # обязательные доп. ключи
        pd["book_id"] = self.book_id

        # --- legacy aliases для старых скриптов ---
        aliases = {
            "raw":            "raw_dir",
            "extracted":      "reader_dir",
            "extracted_dir":  "reader_dir",
            "preprocessed":   "preprocess_dir",
            "preprocessed_dir": "preprocess_dir",
            "ner_outputs":    "ner_dir",
            "postprocessed":  "coref_dir",
            "coref_outputs":  "coref_dir",
        }
        for old, new in aliases.items():
            pd[old] = pd[new]
        # ------------------------------------------

        return pd


def import_stage_module(stage: str):
    mod_path = STAGE_MODULES.get(stage, stage)
    try:
        return importlib.import_module(mod_path)
    except ModuleNotFoundError as e:
        logger.error(f"Не найден модуль для стадии '{stage}': {mod_path}")
        raise e


def run_stage(stage: str, paths: Dict[str, Path], cfg: Dict[str, Any]) -> None:
    logger.info(f"=== [{stage}] start ===")
    module = import_stage_module(stage)
    if not hasattr(module, "run_stage"):
        raise AttributeError(f"Модуль {module.__name__} не содержит run_stage(paths, cfg)")
    module.run_stage(paths, cfg)
    logger.info(f"=== [{stage}] done ===")


def select_stages(order: List[str],
                  only: Optional[List[str]],
                  skip_until: Optional[str],
                  stop_after: Optional[str]) -> List[str]:
    result = list(order)
    if only:
        return [s for s in order if s in only]
    if skip_until:
        if skip_until not in order:
            raise ValueError(f"--skip-until '{skip_until}' не в STAGE_ORDER")
        idx = order.index(skip_until)
        result = order[idx + 1:]
    if stop_after:
        if stop_after not in order:
            raise ValueError(f"--stop-after '{stop_after}' не в STAGE_ORDER")
        idx = order.index(stop_after)
        result = [s for s in result if order.index(s) <= idx]
    return result


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run book processing pipeline")
    p.add_argument("--book-id", required=True)
    p.add_argument("--only", nargs="+")
    p.add_argument("--skip-until")
    p.add_argument("--stop-after")
    p.add_argument("--force", action="store_true")
    p.add_argument("--cfg", type=str)
    p.add_argument("--list", action="store_true")
    return p.parse_args(argv)


def main(argv: List[str]) -> None:
    args = parse_args(argv)

    cfg = {k: (v.copy() if isinstance(v, dict) else v) for k, v in STAGE_CFG.items()}
    if args.cfg:
        ext = load_external_cfg(Path(args.cfg))
        deep_update(cfg, ext)

    logger.remove()
    logger.add(sys.stderr, level=cfg["global"].get("log_level", "INFO"))

    if args.list:
        logger.info("STAGE_ORDER: " + " -> ".join(STAGE_ORDER))
        return

    stages_to_run = select_stages(
        order=STAGE_ORDER,
        only=args.only,
        skip_until=args.skip_until,
        stop_after=args.stop_after,
    )
    if not stages_to_run:
        logger.warning("Нечего запускать.")
        return

    pm = PathManager(Path(cfg["global"]["workspace_root"]), args.book_id)
    paths = pm.paths_dict()

    cfg["global"]["book_id"] = args.book_id
    cfg["global"]["force"] = bool(args.force)

    logger.info(f"Книга: {args.book_id}")
    logger.info("Стадии: " + " -> ".join(stages_to_run))

    for stage in stages_to_run:
        stage_cfg = cfg.get(stage, {})
        stage_cfg.setdefault("force", args.force)
        run_stage(stage, paths, stage_cfg)


if __name__ == "__main__":
    main(sys.argv[1:])
