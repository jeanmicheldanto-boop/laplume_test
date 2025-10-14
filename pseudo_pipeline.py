# -*- coding: utf-8 -*-
"""
Pseudo-pipeline EXT (version corrigée avec logging et métriques) :
- Pseudonymisation réversible : PHONE, NIR, Dates, Adresses (placeholders + mapping JSON)
- NER (optionnel) + Règles externes (YAML/CSV)
- Résolveur de spans (priorités + absorption ETAB + ETAB vs ETAB garde le plus long)
- Pseudonymisation entités (PER/ORG/ETAB/LOC) -> placeholders
- Logging + métriques + validations spans
- PATCHS:
  * protect_placeholders : idempotence si le texte contient déjà des <TOKENS>
  * tokenize_phones(), tokenize_nir() AVANT dates/adresses
  * Dates : regex strictes (pas de faux positifs type 06.92.34)
  * ORG étendues sur "de|d’|du|des|à|au|aux + <LOC>" et cas collés ORG<LOC>
  * ETAB vs ETAB -> garde le plus long
  * ORG sans subtype -> non remplacées
  * tidy_spaces après remplacements

Usage :
  python pseudo_pipeline_fixed.py \
    --in input.txt \
    --out output.txt \
    --report remplacements.csv \
    --conf pseudo_conf_ext.yaml \
    --map mapping.json \
    --log-level DEBUG
"""

import re, sys, csv, yaml, argparse, json, time, logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
from pathlib import Path

# =============================================================================
# 0) Logging & métriques
# =============================================================================

class PipelineMetrics:
    def __init__(self): self.reset()
    def reset(self):
        self.start_time = time.time()
        self.steps_time = {}
        self.entities_count = defaultdict(int)
        self.spans_stats = {"ner_total": 0, "rules_total": 0, "conflicts_resolved": 0, "final_spans": 0}
        self.replacements_stats = defaultdict(int)
        self.dates_count = 0
        self.addresses_count = 0
        self.errors = []
    def start_step(self, step): self.steps_time[step] = time.time(); logging.info(f"🔄 Début étape: {step}")
    def end_step(self, step):
        if step in self.steps_time:
            d = time.time() - self.steps_time[step]; self.steps_time[step] = d
            logging.info(f"✅ Fin étape: {step} ({d:.2f}s)")
    def add_entities(self, spans: List[Dict], source: str):
        for s in spans: self.entities_count[f"{source}_{s.get('label','UNKNOWN')}"] += 1
    def add_error(self, err: str): self.errors.append(err); logging.error(f"❌ Erreur: {err}")
    def print_summary(self):
        total = time.time() - self.start_time
        logging.info("="*60); logging.info("📊 RÉSUMÉ DES MÉTRIQUES"); logging.info("="*60)
        logging.info(f"⏱️  Temps total: {total:.2f}s")
        logging.info("\n🔍 Détection d'entités:")
        for k,v in self.entities_count.items(): logging.info(f"  - {k}: {v}")
        logging.info(f"\n📈 Spans:")
        logging.info(f"  - NER total: {self.spans_stats['ner_total']}")
        logging.info(f"  - Règles total: {self.spans_stats['rules_total']}")
        logging.info(f"  - Conflits résolus: {self.spans_stats['conflicts_resolved']}")
        logging.info(f"  - Spans finaux: {self.spans_stats['final_spans']}")
        logging.info(f"\n🔄 Remplacements:")
        for k,v in self.replacements_stats.items(): logging.info(f"  - {k}: {v}")
        logging.info(f"\n📅 Données sensibles:")
        logging.info(f"  - Dates pseudonymisées: {self.dates_count}")
        logging.info(f"  - Adresses/NIR/Phones tokenisés: {self.addresses_count}")
        if self.errors:
            logging.warning(f"\n⚠️  Erreurs rencontrées: {len(self.errors)}")
            for e in self.errors[:5]: logging.warning(f"  - {e}")

def setup_logging(level: str = "INFO"):
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=lvl, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logging.info(f"🚀 Logging configuré au niveau {level}")

# =============================================================================
# 1) Pseudonymisation réversible (PHONE, NIR, Dates, Adresses)
# =============================================================================

# Téléphone français (01.. à 09.., séparateurs espace/point/tiret)
RE_PHONE = re.compile(r"\b0[1-9](?:[\s\.\-]?\d{2}){4}\b")
# NIR (numéro sécu) : 13 chiffres + 2 clefs, avec espaces possibles
RE_NIR = re.compile(r"\b[12]\s?\d{2}\s?\d{2}\s?\d{2}\s?\d{3}\s?\d{3}\s?\d{2}\b")

# Dates (strictes)
RE_DATE_DMY = re.compile(r"\b(\d{1,2})([/-])(\d{1,2})\2(\d{4})\b")     # JJ/MM/AAAA ou JJ-MM-AAAA
RE_DATE_YMD = re.compile(r"\b(\d{4})([/-])(\d{1,2})\2(\d{1,2})\b")     # AAAA-MM-JJ
RE_DATE_DMY_DOTS = re.compile(r"\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b")    # JJ.MM.AAAA
RE_DATE_FR = re.compile(r"\b(\d{1,2})\s+(janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\s+(\d{4})\b", re.IGNORECASE)

# Adresses (MVP)
RE_STREET = re.compile(
    r"\b(\d{1,4})\s+(?:bis\s+|ter\s+)?(?:rue|avenue|av\.?|bd|boulevard|all(?:ée|\.?)|chemin|impasse|route|place|square|passage)\s+([A-Za-zÀ-ÖØ-öø-ÿ0-9'’\-\s]{2,30})",
    re.IGNORECASE
)
RE_ZIP = re.compile(r"\b\d{5}\b")
PREP_CITY = re.compile(r"\b(?:à|au|aux|dans|sur)\s+([A-ZÉÈÀÂÎÙÔÇ][A-Za-zÀ-ÖØ-öø-ÿ'’\-]+(?:\s+[A-ZÉÈÀÂÎÙÔÇ][A-Za-zÀ-ÖØ-öø-ÿ'’\-]+){0,2})\b")

PLACEHOLDER = re.compile(r"<[A-Z_]+(?:_L\d+)?>")

@dataclass
class PseudonymStore:
    forward: Dict[str, str] = field(default_factory=dict)
    reverse: Dict[str, str] = field(default_factory=dict)
    counters: Dict[str, int] = field(default_factory=dict)
    def _next_id(self, prefix: str) -> int:
        n = self.counters.get(prefix, 0) + 1; self.counters[prefix] = n; return n
    def token_for(self, value: str, prefix: str) -> str:
        if not value or not value.strip(): logging.warning(f"⚠️  Valeur vide pour tokenization: '{value}'"); return value
        if value in self.reverse: return f"<{self.reverse[value]}>"
        idx = self._next_id(prefix); key = f"{prefix}_L{idx}"
        self.forward[key] = value; self.reverse[value] = key
        logging.debug(f"🔤 Nouveau token: {value} -> <{key}>"); return f"<{key}>"
    # ---- Phones & NIR ----
    def tokenize_phones(self, text: str, metrics) -> str:
        def repl(m): metrics.addresses_count += 1; return self.token_for(m.group(0), "PHONE")
        return RE_PHONE.sub(repl, text)
    def tokenize_nir(self, text: str, metrics) -> str:
        def repl(m): metrics.addresses_count += 1; return self.token_for(m.group(0), "NIR")
        return RE_NIR.sub(repl, text)
    # ---- Dates ----
    def tokenize_dates(self, text: str, metrics) -> str:
        def repl(m): metrics.dates_count += 1; return self.token_for(m.group(0), "DATE")
        text = RE_DATE_DMY.sub(repl, text)
        text = RE_DATE_YMD.sub(repl, text)
        text = RE_DATE_DMY_DOTS.sub(repl, text)
        text = RE_DATE_FR.sub(repl, text)
        logging.info(f"📅 Dates tokenisées: {metrics.dates_count}"); return text
    # ---- Adresses ----
    def tokenize_addresses(self, text: str, tokenize_cities: bool, metrics) -> str:
        def repl_street(m): metrics.addresses_count += 1; return self.token_for(m.group(0), "ADDR_STREET")
        def repl_zip(m): metrics.addresses_count += 1; return self.token_for(m.group(0), "ADDR_ZIP")
        text = RE_STREET.sub(repl_street, text); text = RE_ZIP.sub(repl_zip, text)
        if tokenize_cities:
            def repl_city(m):
                city = m.group(1); tok = self.token_for(city, "ADDR_CITY").strip("<>")
                metrics.addresses_count += 1; return m.group(0).replace(city, tok)
            text = PREP_CITY.sub(repl_city, text)
        logging.info(f"🏠 Adresses/NIR/Phones tokenisés: {metrics.addresses_count}"); return text
    # ---- De-pseudo ----
    def depseudonymize(self, text: str) -> str:
        for key in sorted(self.forward.keys(), key=len, reverse=True):
            text = text.replace(f"<{key}>", self.forward[key])
        return text
    def save(self, path: str):
        Path(path).write_text(json.dumps({"forward": self.forward, "counters": self.counters}, ensure_ascii=False, indent=2), encoding="utf-8")
        logging.info(f"💾 Mapping sauvegardé: {path}")
    @classmethod
    def load(cls, path: str) -> "PseudonymStore":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        ps = cls(); ps.forward = data.get("forward", {}); ps.counters = data.get("counters", {}); ps.reverse = {v:k for k,v in ps.forward.items()}
        logging.info(f"📂 Mapping chargé: {path}"); return ps

def protect_placeholders(text: str) -> str:
    text = re.sub(r"([A-Za-zÀ-ÖØ-öø-ÿ])<", r"\1 <", text)  # "Mot<" -> "Mot <"
    text = re.sub(r">(?!\s)", "> ", text)                   # ">Mot" -> "> Mot"
    return text

# =============================================================================
# 2) Configuration
# =============================================================================

@dataclass
class Config:
    ner_model: Optional[str] = None
    rules_path: Optional[str] = None
    gazetteers: Dict[str, str] = None
    thresholds: Dict[str, Any] = None
    date_policy: Dict[str, Any] = None
    address_policy: Dict[str, Any] = None
    local_id_suffix_categories: List[str] = None
    iou_threshold: float = 0.3
    overlap_strategy: str = "priority"

def load_conf(path: str) -> Config:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    data.setdefault("iou_threshold", 0.3)
    data.setdefault("overlap_strategy", "priority")
    cfg = Config(**data); logging.info(f"⚙️  Configuration chargée: {path}"); return cfg

# =============================================================================
# 3) Validation spans
# =============================================================================

def validate_span(span: Dict[str, Any], text_len: int) -> bool:
    try:
        s,e = span.get("start",-1), span.get("end",-1)
        return isinstance(s,int) and isinstance(e,int) and 0 <= s < e <= text_len
    except Exception: return False

def validate_and_filter_spans(spans: List[Dict[str, Any]], text_len: int, metrics: PipelineMetrics) -> List[Dict[str, Any]]:
    out = []
    for sp in spans:
        if validate_span(sp, text_len): out.append(sp)
        else: metrics.add_error(f"Span invalide: {sp}")
    return out

# =============================================================================
# 4) NER (optionnel)
# =============================================================================

def build_ner(ner_model: Optional[str] = None, metrics: PipelineMetrics = None):
    if ner_model:
        try:
            logging.info(f"🤖 Chargement modèle NER avec gestion chunks: {ner_model}")
            from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
            tok = AutoTokenizer.from_pretrained(ner_model, use_fast=True)
            mdl = AutoModelForTokenClassification.from_pretrained(ner_model)
            ner = pipeline("token-classification", model=mdl, tokenizer=tok, aggregation_strategy="simple")
            
            def detect_chunked(text: str) -> List[Dict[str, Any]]:
                """Détection NER avec découpage en chunks pour éviter la troncature à 512 tokens."""
                # Tokeniser pour vérifier la longueur
                tokens = tok.encode(text, add_special_tokens=True)
                
                if len(tokens) <= 450:  # Marge de sécurité sous 512
                    # Texte court, traitement normal
                    out = []
                    for e in ner(text):
                        g = e.get("entity_group","")
                        if g in {"ORG","PER","LOC"}:
                            out.append({"start": int(e["start"]), "end": int(e["end"]), "text": text[int(e["start"]):int(e["end"])],
                                        "label": g, "source": "NER", "score": float(e.get("score",0.0)), "subtype": None, "fields": {}})
                    logging.debug(f"🤖 NER court détecté: {len(out)} entités")
                    return out
                
                # Texte long, découpage en chunks
                logging.info(f"📏 Texte long ({len(tokens)} tokens), découpage en chunks")
                sentences = re.split(r'(?<=[.!?])\s+(?=\d+\.|\S)', text)
                all_entities = []
                current_pos = 0
                
                i = 0
                while i < len(sentences):
                    # Construire un chunk d'environ 400 tokens
                    chunk_sentences = []
                    chunk_tokens = 0
                    
                    while i < len(sentences) and chunk_tokens < 400:
                        sentence = sentences[i]
                        sentence_tokens = len(tok.encode(sentence, add_special_tokens=False))
                        
                        if chunk_tokens + sentence_tokens > 450 and chunk_sentences:
                            break
                        
                        chunk_sentences.append(sentence)
                        chunk_tokens += sentence_tokens
                        i += 1
                    
                    if not chunk_sentences:
                        chunk_sentences = [sentences[i]]
                        i += 1
                    
                    chunk_text = ' '.join(chunk_sentences)
                    chunk_start = current_pos
                    
                    # Traiter le chunk
                    try:
                        for e in ner(chunk_text):
                            g = e.get("entity_group","")
                            if g in {"ORG","PER","LOC"}:
                                global_start = chunk_start + int(e["start"])
                                global_end = chunk_start + int(e["end"])
                                all_entities.append({
                                    "start": global_start, 
                                    "end": global_end, 
                                    "text": text[global_start:global_end],
                                    "label": g, 
                                    "source": "NER_CHUNK", 
                                    "score": float(e.get("score",0.0)), 
                                    "subtype": None, 
                                    "fields": {}
                                })
                    except Exception as ex:
                        logging.warning(f"⚠️  Erreur traitement chunk: {ex}")
                    
                    # Mise à jour position
                    current_pos += len(chunk_text) + 1  # +1 pour l'espace entre chunks
                
                logging.debug(f"🧩 NER chunks détecté: {len(all_entities)} entités")
                return all_entities
            
            logging.info("✅ Modèle NER avec chunks chargé")
            return detect_chunked
        except Exception as ex:
            msg = f"Transformers indisponible ({ex}); fallback regex-only"; logging.warning(f"⚠️  {msg}")
            if metrics: metrics.add_error(msg)
    logging.info("🔧 Utilisation du fallback regex pour NER")
    ORG_HINT = re.compile(r"\b(CAF|CPAM|ARS|URSSAF|Pôle emploi|Croix-Rouge française)\b", re.I)
    LOC_HINT = re.compile(r"\b(Paris|Lyon|Marseille|Bordeaux|Lille|Nantes|Rennes|Toulouse|Nice)\b", re.I)
    def detect_regex(text: str) -> List[Dict[str, Any]]:
        spans = []
        for m in ORG_HINT.finditer(text):
            spans.append({"start": m.start(), "end": m.end(), "text": m.group(), "label": "ORG","source": "NER_FALLBACK","score":0.5,"subtype":None,"fields":{}})
        for m in LOC_HINT.finditer(text):
            spans.append({"start": m.start(), "end": m.end(), "text": m.group(), "label": "LOC","source": "NER_FALLBACK","score":0.5,"subtype":None,"fields":{}})
        logging.debug(f"🔧 Fallback NER: {len(spans)} entités"); return spans
    return detect_regex

# =============================================================================
# 5) Règles externes
# =============================================================================

def load_rules(path: str, metrics: PipelineMetrics):
    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        org_conf = data.get("org_regex", {}); etab_conf = data.get("etab_categories", {}); patterns = data.get("patterns", {})
        org_regex, etab_regex = {}, {}
        for cat, pats in org_conf.items():
            compiled = []
            for pat in pats:
                try: compiled.append(re.compile(pat, re.I))
                except Exception as e: metrics.add_error(f"Pattern regex invalide '{pat}': {e}")
            org_regex[cat] = compiled
        for cat, pats in etab_conf.items():
            compiled = []
            for pat in pats:
                try: compiled.append(re.compile(pat, re.I))
                except Exception as e: metrics.add_error(f"Pattern regex invalide '{pat}': {e}")
            etab_regex[cat] = compiled
        loc_hint = re.compile(patterns["loc_hint"], re.I) if patterns.get("loc_hint") else None
        etab_name_window = re.compile(patterns["etab_name_window"], re.I) if patterns.get("etab_name_window") else None
        logging.info(f"📋 Règles chargées: {len(org_regex)} ORG, {len(etab_regex)} ETAB")
        return org_regex, etab_regex, loc_hint, etab_name_window
    except Exception as e:
        metrics.add_error(f"Erreur chargement règles: {e}")
        logging.error(f"❌ Erreur chargement règles: {e}")
        return {}, {}, None, None

def load_gazetteers(conf: Dict[str,str], metrics: PipelineMetrics):
    gaz = {}
    for key, path in (conf or {}).items():
        try:
            p = Path(path); rows = []
            if p.exists():
                with open(p, "r", encoding="utf-8") as f:
                    rows = [r for r in csv.DictReader(f)]
                logging.info(f"📚 Gazetteer '{key}' chargé: {len(rows)} entrées")
            else:
                logging.warning(f"⚠️  Gazetteer non trouvé: {path}")
            gaz[key] = rows
        except Exception as e:
            metrics.add_error(f"Erreur gazetteer {key}: {e}"); gaz[key] = []
    return gaz

# =============================================================================
# 6) Détection par règles
# =============================================================================

def detect_rules(text: str, org_regex, etab_regex, loc_hint, etab_name_window, metrics: PipelineMetrics) -> List[Dict[str, Any]]:
    spans = []
    # ORG
    for subtype, regs in org_regex.items():
        for rx in regs:
            try:
                for m in rx.finditer(text):
                    spans.append({"start": m.start(),"end": m.end(),"text": m.group(),"label":"ORG","source":"RULE","score":0.95,"subtype":subtype,"fields":{}})
            except Exception as e: metrics.add_error(f"Erreur regex ORG {subtype}: {e}")
    # ETAB
    for cat, regs in etab_regex.items():
        for rx in regs:
            try:
                for m in rx.finditer(text):
                    start, end = m.start(), m.end()
                    window = text[end:end+120]; span_start, span_end = start, end
                    fields = {"category": cat, "name": None, "locality": None}
                    if etab_name_window:
                        try:
                            m_int = etab_name_window.search(window)
                            if m_int: span_end = end + m_int.end(); fields["name"] = m_int.group(1).strip()
                        except Exception as e: metrics.add_error(f"Erreur etab_name_window: {e}")
                    if loc_hint:
                        try:
                            m_loc = loc_hint.search(window)
                            if m_loc: span_end = max(span_end, end + m_loc.end()); fields["locality"] = m_loc.group(0)
                        except Exception as e: metrics.add_error(f"Erreur loc_hint: {e}")
                    spans.append({"start": span_start,"end": span_end,"text": text[span_start:span_end],
                                  "label":"ETAB","source":"RULE","score":0.90,"subtype":cat,"fields":fields})
            except Exception as e: metrics.add_error(f"Erreur regex ETAB {cat}: {e}")
    logging.debug(f"📋 Règles détectées: {len(spans)} entités"); return spans

# =============================================================================
# 7) PATCH: extension ORG avec localité à droite (inclut 'à/au/aux' + cas collés)
# =============================================================================

PREP_ANY = re.compile(r"\s*(?:de|d’|d'|du|des|à|au|aux)\s+", re.IGNORECASE)

def extend_org_with_loc(text: str, spans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    loc_by_start = {s["start"]: s for s in spans if s["label"] == "LOC"}
    extended = []
    for s in spans:
        if s["label"] != "ORG":
            extended.append(s); continue
        end = s["end"]
        # cas collé ORG<LOC>
        if end < len(text) and text[end:end+1] == '<':
            loc = loc_by_start.get(end)
            if loc:
                s = dict(s); s["end"] = loc["end"]; extended.append(s); continue
        # prépositions
        m = PREP_ANY.match(text[end:end+10] or "")
        if m:
            after = end + m.end()
            loc = loc_by_start.get(after)
            if loc:
                s = dict(s); s["end"] = loc["end"]
        extended.append(s)
    return extended

# =============================================================================
# 8) Résolution de spans (priorités + ETAB vs ETAB)
# =============================================================================

def iou(a: Tuple[int,int], b: Tuple[int,int]) -> float:
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1]-a[0]) + (b[1]-b[0]) - inter
    return inter/union if union > 0 else 0.0

def resolve_spans(spans: List[Dict[str, Any]], conf: Config, metrics: PipelineMetrics) -> List[Dict[str, Any]]:
    def priority(s):
        base = 0
        if s["source"] == "RULE" and s["label"] == "ETAB": base = 100
        elif s["source"] == "RULE" and s["label"] == "ORG": base = 90
        elif s["source"].startswith("NER") and s["label"] == "ORG": base = 70
        elif s["source"].startswith("NER") and s["label"] == "LOC": base = 60
        else: base = 50
        if s.get("subtype") in {"ORG_CAF","ORG_CPAM","ORG_PUBLIC_TERR","ORG_BAILLEUR_SOC","ORG_CCAS_CIAS","ORG_FINANCIER"}:
            base += 5
        return base + float(s.get("score",0))

    spans_sorted = sorted(spans, key=priority, reverse=True)
    accepted: List[Dict[str, Any]] = []
    conflicts = 0

    for cand in spans_sorted:
        keep = True; to_remove = []
        for acc in accepted:
            ov = iou((cand["start"], cand["end"]), (acc["start"], acc["end"]))
            if ov >= conf.iou_threshold:
                conflicts += 1
                # ETAB vs ETAB -> garder le plus long
                if cand["label"] == "ETAB" and acc["label"] == "ETAB":
                    len_c = cand["end"]-cand["start"]; len_a = acc["end"]-acc["start"]
                    if len_c >= len_a: to_remove.append(acc)
                    else: keep = False
                    continue
                # Absorption ETAB (cand englobe acc)
                if cand["label"] == "ETAB" and cand["start"] <= acc["start"] and cand["end"] >= acc["end"]:
                    to_remove.append(acc); continue
                # Stratégie priorité
                if conf.overlap_strategy == "priority":
                    if priority(acc) >= priority(cand): keep = False; break
                    else: to_remove.append(acc)
        for acc in to_remove:
            if acc in accepted: accepted.remove(acc)
        if keep: accepted.append(cand)

    # Nettoyage: supprimer ce qui est inclus dans un ETAB
    etabs = [s for s in accepted if s["label"]=="ETAB"]
    cleaned = []
    for s in accepted:
        if s["label"]!="ETAB" and any(e["start"] <= s["start"] and e["end"] >= s["end"] for e in etabs):
            conflicts += 1; continue
        cleaned.append(s)

    metrics.spans_stats["conflicts_resolved"] = conflicts
    logging.info(f"🔀 Conflits résolus: {conflicts}, spans finaux: {len(cleaned)}")
    return sorted(cleaned, key=lambda x: x["start"])

# =============================================================================
# 9) Pseudonymisation entités + nettoyage
# =============================================================================

def make_local_counters(): return defaultdict(lambda: 1)
def next_local_id(counters: Dict[str,int], cat: str) -> int: n=counters[cat]; counters[cat]=n+1; return n

def pseudo_replace_entities(text: str, entities: List[Dict[str, Any]], conf: Config, metrics: PipelineMetrics) -> Tuple[str, List[Dict[str, Any]]]:
    out = text; counters = make_local_counters(); repl = []
    for ent in sorted(entities, key=lambda x: x["start"], reverse=True):
        try:
            label = ent["label"]; subtype = ent.get("subtype"); placeholder = None
            if label == "ORG":
                # Pas de remplacement si subtype inconnu (évite <ORG_PUBLIC_NAT>)
                if not subtype: continue
                cat = "ORG_ASSOC_LOCAL" if subtype == "ORG_ASSOC_GENERIC" else subtype
                if cat in set((conf.local_id_suffix_categories or [])):
                    placeholder = f"<{cat}_L{next_local_id(counters, cat)}>"
                else:
                    placeholder = f"<{cat}>"
            elif label == "ETAB":
                cat = ent.get("fields",{}).get("category", subtype) or "ETAB"
                placeholder = f"<ETAB_{cat}_L{next_local_id(counters, f'ETAB_{cat}')}>" 
            elif label == "PER":
                placeholder = "<PER>"
            elif label == "LOC":
                placeholder = "<LOC>"
            if placeholder:
                out = out[:ent["start"]] + placeholder + out[ent["end"]:]
                repl.append({"from": ent["text"], "to": placeholder, "label": label, "subtype": subtype})
                metrics.replacements_stats[label] += 1
                logging.debug(f"🔄 {ent['text']} -> {placeholder}")
        except Exception as e:
            metrics.add_error(f"Erreur remplacement entité: {e}")
    logging.info(f"🔄 Entités remplacées: {len(repl)}"); 
    return out, list(reversed(repl))

def tidy_spaces(text: str) -> str:
    text = re.sub(r">\s+<", "> <", text)
    text = re.sub(r"([^\s])<", r"\1 <", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = text.replace(" .", ".").replace(" ,", ",")
    return text

# =============================================================================
# 10) Orchestration principale
# =============================================================================

def process_text(text: str, conf: Config, save_mapping_path: Optional[str] = None) -> Dict[str, Any]:
    metrics = PipelineMetrics()

    # Pré-normalisation (idempotence)
    text = protect_placeholders(text)

    # A) Pseudonymisation sensible AVANT NER/Règles (ordre : téléphone -> NIR -> dates -> adresses)
    metrics.start_step("PSEUDO_SENSITIVE_PRE")
    ps = PseudonymStore()
    work_text = ps.tokenize_phones(text, metrics)
    work_text = ps.tokenize_nir(work_text, metrics)
    work_text = ps.tokenize_dates(work_text, metrics)
    tokenize_cities = bool((conf.address_policy or {}).get("tokenize_cities", False))
    work_text = ps.tokenize_addresses(work_text, tokenize_cities, metrics)
    metrics.end_step("PSEUDO_SENSITIVE_PRE")

    # 1) NER
    metrics.start_step("NER")
    ner_detect = build_ner(conf.ner_model, metrics)
    spans_ner = ner_detect(work_text)
    spans_ner = validate_and_filter_spans(spans_ner, len(work_text), metrics)
    metrics.spans_stats["ner_total"] = len(spans_ner); metrics.add_entities(spans_ner, "NER")
    metrics.end_step("NER")

    # 2) Règles
    metrics.start_step("RULES")
    org_regex, etab_regex, loc_hint, etab_name_window = load_rules(conf.rules_path, metrics)
    _gaz = load_gazetteers(conf.gazetteers, metrics)
    spans_rule = detect_rules(work_text, org_regex, etab_regex, loc_hint, etab_name_window, metrics)
    spans_rule = validate_and_filter_spans(spans_rule, len(work_text), metrics)
    metrics.spans_stats["rules_total"] = len(spans_rule); metrics.add_entities(spans_rule, "RULES")
    metrics.end_step("RULES")

    # 3) Résolution + extension ORG
    metrics.start_step("RESOLUTION")
    spans = resolve_spans(spans_ner + spans_rule, conf, metrics)
    spans = extend_org_with_loc(work_text, spans)
    metrics.spans_stats["final_spans"] = len(spans)
    metrics.end_step("RESOLUTION")

    # 4) Pseudonymisation entités
    metrics.start_step("PSEUDO_ENTITIES")
    pseudo_text, repl_entities = pseudo_replace_entities(work_text, spans, conf, metrics)
    pseudo_text = tidy_spaces(pseudo_text)
    metrics.end_step("PSEUDO_ENTITIES")

    # 5) Sauvegarde mapping
    mapping_path = None
    if save_mapping_path:
        ps.save(save_mapping_path); mapping_path = save_mapping_path

    # 6) Métriques finales
    metrics.print_summary()

    return {
        "original": text,
        "preprocessed": work_text,
        "spans_ner": spans_ner,
        "spans_rule": spans_rule,
        "spans_final": spans,
        "replacements": repl_entities,
        "result": pseudo_text,
        "mapping_path": mapping_path,
        "metrics": metrics
    }

# =============================================================================
# 11) CLI
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="Pipeline de pseudonymisation avancé avec logging")
    ap.add_argument("--in", dest="inp", required=True, help="Fichier texte d'entrée")
    ap.add_argument("--out", dest="out", required=True, help="Fichier texte pseudonymisé")
    ap.add_argument("--report", dest="rep", help="CSV des remplacements entités")
    ap.add_argument("--conf", dest="conf", required=True, help="Fichier YAML de configuration")
    ap.add_argument("--map", dest="map_path", help="JSON mapping réversible (sensible)")
    ap.add_argument("--log-level", dest="log_level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Niveau de logging")
    args = ap.parse_args()

    setup_logging(args.log_level)

    try:
        logging.info(f"📂 Chargement fichier: {args.inp}")
        conf = load_conf(args.conf)
        text = Path(args.inp).read_text(encoding="utf-8")
        logging.info(f"📄 Texte chargé: {len(text)} caractères")

        logging.info("🚀 Début du traitement")
        res = process_text(text, conf, save_mapping_path=args.map_path)

        Path(args.out).write_text(res["result"], encoding="utf-8")
        logging.info(f"💾 Résultat sauvegardé: {args.out}")

        if args.rep:
            with open(args.rep, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["from","to","label","subtype"])
                w.writeheader()
                for r in res["replacements"]:
                    w.writerow(r)
            logging.info(f"📊 Rapport sauvegardé: {args.rep}")

        logging.info("🎉 PSEUDONYMISATION TERMINÉE AVEC SUCCÈS")

    except Exception as e:
        logging.error(f"💥 Erreur fatale: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
