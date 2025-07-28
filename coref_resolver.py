# -*- coding: utf-8 -*-
"""
Стадия Coref: связывание местоимений с именованными упоминаниями персонажей.
Чтение:
  30_ner/<book_id>_ner.json
  30_ner/<book_id>_mentions_index.json
  20_preprocessed/<book_id>_preprocessed.json
Запись:
  50_coref/<book_id>_coref.json
  50_coref/<book_id>_mentions_index.json

Основные улучшения:
- Параметризованный размер окна и возможность связывать кросс-сценовые упоминания
- Конфигурируемые словари местоимений с указанием пола
- Опциональное использование нейросетевого резольвера кореференции (neuralcoref)
- Подробное логирование количества разрешённых и неразрешённых местоимений
- Добавление поля `type="pronoun"` для упоминаний-местоимений
- Автоматическое сопоставление нейросетевых кластеров с entity_id через алиасы
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Tuple, DefaultDict
from collections import defaultdict
import json
import re

from loguru import logger


def run_stage(paths: Dict[str, Path], cfg: Dict[str, Any]) -> None:
    # Идентификатор книги
    book_id = paths['book_root'].name

    # Пути к входным данным
    ner_path = paths['ner_dir'] / f"{book_id}_ner.json"
    idx_path = paths['ner_dir'] / f"{book_id}_mentions_index.json"
    prep_path = paths['preprocess_dir'] / f"{book_id}_preprocessed.json"

    # Пути к выходным данным
    out_dir = paths['coref_dir']; out_dir.mkdir(parents=True, exist_ok=True)
    coref_path = out_dir / f"{book_id}_coref.json"
    coref_idx_path = out_dir / f"{book_id}_mentions_index.json"

    # Пропускаем, если уже есть
    if coref_path.exists() and coref_idx_path.exists() and not cfg.get('force', False):
        logger.info(f"[coref] Есть результаты, пропускаю: {coref_path.name}")
        return

    # Проверка наличия
    if not ner_path.exists(): raise FileNotFoundError(f"[coref] Нет NER: {ner_path}")
    if not prep_path.exists(): raise FileNotFoundError(f"[coref] Нет препроцессинга: {prep_path}")

    # Загрузка исходных JSON
    with ner_path.open('r', encoding='utf-8') as f: ner_data = json.load(f)
    with idx_path.open('r', encoding='utf-8') as f: mentions_index = json.load(f)
    with prep_path.open('r', encoding='utf-8') as f: prep_data = json.load(f)

    # Параметры
    window = cfg.get('window', 3)
    cross_scene = cfg.get('cross_scene', False)
    attach_type = cfg.get('attach_type_field', True)
    pronouns_cfg = cfg.get('pronouns', {})
    use_neural = cfg.get('use_neural_coref', False)

    # Настройка нейросетевого резольвера
    neural_model = None
    if use_neural:
        try:
            import spacy
            from neuralcoref import Coref
            nlp = spacy.load('ru_core_news_lg')
            neural_model = Coref(nlp)
        except Exception:
            logger.warning("Не удалось загрузить neuralcoref, отключаем его.")
            use_neural = False

    # Маппинг местоимение -> пол
    pronoun2gender = _build_pronoun_map(pronouns_cfg)

    # Подготовка сущностей и алиасов для нейро-функции
    entities = ner_data['entities']; _ensure_entity_lists(entities)
    alias_map: Dict[str,int] = {}
    for ent in entities:
        eid = ent['id']
        for alias in ent.get('aliases', []): alias_map[alias.lower()] = eid

    # Глобальный счётчик упоминаний
    max_mid = _find_max_mention_number(entities)
    mention_counter = max_mid + 1

    # Индекс именованных упоминаний
    name_map: DefaultDict[str,List[Tuple[int,str]]] = defaultdict(list)
    for ent in entities:
        for m in ent['mentions']:
            if m.get('type') != 'name': continue
            key = _key(m['chapter'], m['scene'], m['sent_id'], cross_scene)
            name_map[key].append((ent['id'], m['mention_id']))

    # Собираем местоимения
    pronoun_mentions: List[Dict[str,Any]] = []
    resolved = 0; unresolved = 0

    for ch in prep_data['chapters']:
        for sc in ch['scenes']:
            # Собираем текст сцены и массив предложений
            scene_text = sc.get('text') or " ".join(s['text'] for s in sc['sentences'])
            sent_map = {s['id']:(s['text'], _offsets_in_scene(sc['sentences'], s['id'])) for s in sc['sentences']}
            # Если нейросетевой, готовим doc
            doc = neural_model(scene_text) if use_neural else None

            for s in sc['sentences']:
                text, (base_offset, _) = sent_map[s['id']]
                for token, start, end, gender in _find_pronouns(text, pronoun2gender):
                    # Определяем target через нейро или правило
                    if use_neural:
                        target = _neural_resolve_neural(doc, base_offset+start, alias_map)
                    else:
                        target = _resolve_pronoun_simple(ch['id'], sc['id'], s['id'], gender,
                                                         window, name_map, entities)
                    if target is None:
                        unresolved += 1; continue
                    resolved += 1
                    m_id = f"ent_{target}:m_{mention_counter}"; mention_counter+=1
                    pm = {'mention_id':m_id,'entity_id':target,'chapter':ch['id'],
                          'scene':sc['id'],'sent_id':s['id'],'start':start,'end':end,
                          'text':token,'type':'pronoun' if attach_type else None,
                          'source':'neural_coref' if use_neural else 'coref_rule'}
                    pronoun_mentions.append(pm)
                    key=_key(ch['id'],sc['id'],s['id'],cross_scene)
                    mentions_index.setdefault(key,[]).append(m_id)

    logger.info(f"[coref] Разрешено: {resolved}, неразрешено: {unresolved}")

    # Добавляем и сортируем
    for pm in pronoun_mentions:
        entities[pm['entity_id']]['mentions'].append(pm)
    for ent in entities:
        ent['mentions'].sort(key=lambda m:(m['chapter'],m['scene'],m['sent_id'],m.get('start',-1)))

    # Сохраняем
    with coref_path.open('w',encoding='utf-8') as f:
        json.dump({'book_id':book_id,'entities':entities},f,ensure_ascii=False,indent=2)
    with coref_idx_path.open('w',encoding='utf-8') as f:
        json.dump(mentions_index,f,ensure_ascii=False,indent=2)
    logger.info(f"[coref] Сохранено: {coref_path.name}")


# --- Хелперы ---

def _offsets_in_scene(sents: List[Dict[str,Any]], sid:int) -> Tuple[int,int]:
    # Возвращает (base_offset, length) предложения sid в тексте сцены
    pos=0
    for s in sents:
        text=s['text']; length=len(text)
        if s['id']==sid: return pos, length
        pos+=length+1
    return 0,0


def _ensure_entity_lists(entities: List[Dict[str,Any]]):
    for ent in entities: ent.setdefault('mentions', [])


def _key(ch:int,sc:int,sid:int,cs:bool)->str:
    return f"{ch}::ALL::{sid}" if cs else f"{ch}::{sc}::{sid}"


def _build_pronoun_map(cfg:Dict[str,List[str]])->Dict[str,str]:
    mp={};
    for g,lst in cfg.items():
        for p in lst: mp[p.lower()]=g
    return mp


def _find_pronouns(text:str,map_g)->List[Tuple[str,int,int,str]]:
    if not map_g: return []
    pat=r"\b("+"|".join(map(re.escape,map_g))+r")\b"
    return [(m.group(0),m.start(),m.end(),map_g.get(m.group(0).lower(),'neutral')) for m in re.finditer(pat,text,re.IGNORECASE)]


def _resolve_pronoun_simple(ch,sc,sid,gender,window,name_map,entities):
    cands=[]
    for d in range(window+1):
        key=f"{ch}::{sc}::{sid-d}"
        cands+=name_map.get(key,[])
    if not cands: return None
    if gender!='neutral':
        same=[(e,m) for e,m in cands if entities[e].get('gender')==gender]
        if same: return same[-1][0]
    return cands[-1][0]


def _find_max_mention_number(entities:List[Dict[str,Any]])->int:
    mx=-1
    for ent in entities:
        for m in ent.get('mentions',[]):
            mid=m.get('mention_id','');
            if ':m_' in mid:
                try: num=int(mid.split(':m_')[1]); mx=max(mx,num)
                except: pass
    return mx


def _neural_resolve_neural(doc, abs_char, alias_map):
    # Ищем кластер, содержащий местоимение по абсолютному смещению
    for cluster in doc._.coref_clusters:
        for mention in cluster.mentions:
            if mention.start_char <= abs_char < mention.end_char:
                main = cluster.main.text.lower()
                return alias_map.get(main)
    return None
