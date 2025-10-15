#!/usr/bin/env python3
"""
Pipeline de pseudonymisation hybride : NER chunked + Règles avec résolution de conflits
"""
import argparse
import csv
import json
import logging
import re
import sys
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Set
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline, CamembertTokenizer

# Import des nouveaux modules d'amélioration
try:
    from text_processing import TextNormalizer, StopWordsFilter, PriorityMatrix
    ENHANCED_MODULES_AVAILABLE = True
except ImportError:
    # Fallback si les modules ne sont pas disponibles
    TextNormalizer = None
    StopWordsFilter = None
    PriorityMatrix = None
    ENHANCED_MODULES_AVAILABLE = False


def _norm_key(s: str) -> str:
    """Normalise une clé pour l'indexation en supprimant les espaces multiples et normalisant les apostrophes"""
    return re.sub(r"\s+", " ", s.replace("'", "'")).strip()


def _is_establishment_name(text: str, context: str) -> bool:
    """Détermine si un nom ressemble à un établissement basé sur des heuristiques"""
    # Normaliser le texte
    norm_text = _norm_key(text)
    
    # Heuristique 1: Commence par des articles caractéristiques d'établissements
    if norm_text.startswith(("Les ", "Le ", "La ", "L'")):
        # Exclure les écoles générales et les villes connues
        if (not norm_text.lower().startswith(("école", "collège", "lycée")) and
            not any(city in norm_text.lower() for city in [
                "paris", "lyon", "marseille", "toulouse", "nice", "nantes", 
                "montpellier", "strasbourg", "bordeaux", "lille", "rennes", 
                "reims", "nanterre", "bobigny", "créteil", "montreuil"
            ])):
            return True
    
    # Heuristique 2: Contexte contient des verbes indicateurs d'établissement
    establishment_verbs = [
        "accueille", "accompagne", "héberge", "prend en charge", "suit", 
        "oriente", "évalue", "propose", "offre", "dispense", "assure"
    ]
    if any(verb in context for verb in establishment_verbs):
        return True
    
    # Heuristique 3: Noms d'arbres/plantes (souvent utilisés pour les établissements)
    plant_names = [
        "tilleuls", "peupliers", "chênes", "ormes", "platanes", "érables",
        "acacias", "cyprès", "pins", "sapins", "bouleaux", "frênes",
        "roses", "lilas", "jasmin", "glycines", "magnolias"
    ]
    if any(plant in norm_text.lower() for plant in plant_names):
        return True
    
    return False


class PseudonymStore:
    """Gestionnaire de pseudonymisation avec compteurs et sauvegarde"""
    
    def __init__(self):
        self.forward = {}  # original -> pseudonyme
        self.reverse = {}  # pseudonyme -> original
        self.counters = {"PER": 0, "ORG": 0, "LOC": 0, "ETAB": 0, "DATE": 0, "PHONE": 0, "NIR": 0, "TIME": 0, "EMAIL": 0}
    
    def pseudonymize(self, text: str, label: str, category: str = None) -> str:
        """Pseudonymise un texte ou retourne le pseudonyme existant"""
        if text in self.forward:
            return self.forward[text]
        
        # Utiliser la catégorie spécifique si fournie, sinon utiliser le label générique
        pseudo_label = category if category else label
        
        # Créer le compteur pour cette catégorie si nécessaire
        if pseudo_label not in self.counters:
            self.counters[pseudo_label] = 0
            
        self.counters[pseudo_label] += 1
        replacement = f"<{pseudo_label}_{self.counters[pseudo_label]}>"
        self.forward[text] = replacement
        self.reverse[replacement] = text
        return replacement
    
    def save(self, path: str):
        """Sauvegarde le mapping au format JSON"""
        data = {
            "forward": self.forward,
            "reverse": self.reverse,
            "counters": self.counters,
            "stats": {
                "total_entities": len(self.forward),
                "per_type": dict(self.counters)
            }
        }
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logging.info(f"💾 Mapping sauvé: {path}")
    
    @classmethod
    def load(cls, path: str) -> "PseudonymStore":
        """Charge un mapping existant"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        store = cls()
        store.forward = data.get("forward", {})
        store.reverse = data.get("reverse", {})
        store.counters = data.get("counters", {"PER": 0, "ORG": 0, "LOC": 0, "ETAB": 0})
        
        logging.info(f"📂 Mapping chargé: {path}")
        return store
    
    def depseudonymize(self, text: str) -> str:
        """Restaure le texte original à partir des pseudonymes"""
        result = text
        for pseudonym, original in self.reverse.items():
            result = result.replace(pseudonym, original)
        return result


class GazetteerEngine:
    """Moteur de gazetteers pour détecter les entités par correspondance exacte"""
    
    def __init__(self, gazetteer_dir: str):
        self.gazetteer_dir = Path(gazetteer_dir)
        self.gazetteers = {}
        self._load_gazetteers()
    
    def _load_gazetteers(self):
        """Charge tous les fichiers CSV de gazetteers"""
        if not self.gazetteer_dir.exists():
            logging.warning(f"⚠️ Dossier gazetteers introuvable: {self.gazetteer_dir}")
            return
        
        csv_files = list(self.gazetteer_dir.glob("*.csv"))
        if not csv_files:
            logging.warning(f"⚠️ Aucun fichier CSV trouvé dans: {self.gazetteer_dir}")
            return
        
        for csv_file in csv_files:
            gazetteer_name = csv_file.stem
            entries = []
            
            try:
                with open(csv_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if 'name' in row and 'category' in row:
                            entries.append({
                                'name': row['name'].strip(),
                                'category': row['category'].strip()
                            })
                        elif 'keyword' in row and 'category' in row:
                            entries.append({
                                'name': row['keyword'].strip(),
                                'category': row['category'].strip()
                            })
                
                if entries:
                    self.gazetteers[gazetteer_name] = entries
                    logging.info(f"📚 Gazetteer {gazetteer_name}: {len(entries)} entrées")
                
            except Exception as e:
                logging.warning(f"⚠️ Erreur chargement {csv_file}: {e}")
        
        total_entries = sum(len(entries) for entries in self.gazetteers.values())
        logging.info(f"✅ Gazetteers chargés: {len(self.gazetteers)} fichiers, {total_entries} entrées")
    
    def detect_entities(self, text: str) -> List[Dict]:
        """Détecte les entités selon les gazetteers par correspondance exacte"""
        entities = []
        exclusions_count = 0
        
        for gazetteer_name, entries in self.gazetteers.items():
            # Traiter gazetteer_exclusions séparément (ne pas créer d'entités)
            if gazetteer_name == 'gazetteer_exclusions':
                continue
                
            for entry in entries:
                name = entry['name']
                category = entry['category']
                
                # Recherche case-insensitive avec correspondance de mots entiers
                pattern = r'\b' + re.escape(name) + r'\b'
                
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    # Trim le texte détecté
                    matched_text = match.group().strip()
                    
                    # Ignorer si vide après trim ou trop court (< 2 chars sauf codes)
                    if not matched_text or (len(matched_text) < 2 and not category.endswith('_CP')):
                        continue
                    
                    # Déterminer le label final
                    if category.startswith('ORG_'):
                        label = 'ORG'
                    elif category.startswith('ETAB_'):
                        label = 'ETAB'
                    else:
                        label = 'ORG'  # Par défaut
                    
                    entities.append({
                        "start": match.start(),
                        "end": match.end(),
                        "text": matched_text,  # Utiliser le texte trimé
                        "label": label,
                        "source": "GAZETTEER",
                        "category": category,
                        "gazetteer": gazetteer_name,
                        "score": 1.0  # Score maximum pour gazetteers
                    })
        
        logging.info(f"📖 Gazetteers détectés: {len(entities)} entités")
        return entities


class RulesEngine:
    """Moteur de règles pour détecter les organisations, établissements et données sensibles"""
    
    def __init__(self, rules_file: str):
        self.rules_file = rules_file
        self.org_patterns = {}
        self.etab_patterns = {}
        self.date_patterns = {}
        self.phone_patterns = {}
        self.nir_patterns = {}
        self.time_patterns = {}
        self.email_patterns = {}
        self.profession_patterns = {}
        self._load_rules()
    
    def _load_rules(self):
        """Charge les règles depuis le fichier YAML"""
        if not Path(self.rules_file).exists():
            logging.warning(f"⚠️ Fichier de règles introuvable: {self.rules_file}")
            return
        
        with open(self.rules_file, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        # Règles ORG
        org_rules = data.get('org_regex', {})
        for category, patterns in org_rules.items():
            compiled_patterns = []
            for pattern in patterns:
                try:
                    compiled_patterns.append(re.compile(pattern, re.IGNORECASE))
                except re.error as e:
                    logging.warning(f"⚠️ Pattern invalide '{pattern}': {e}")
            
            if compiled_patterns:
                self.org_patterns[category] = compiled_patterns
        
        # Règles ETAB
        etab_rules = data.get('etab_categories', {})
        for category, patterns in etab_rules.items():
            compiled_patterns = []
            for pattern in patterns:
                try:
                    compiled_patterns.append(re.compile(pattern, re.IGNORECASE))
                except re.error as e:
                    logging.warning(f"⚠️ Pattern invalide '{pattern}': {e}")
            
            if compiled_patterns:
                self.etab_patterns[category] = compiled_patterns
        
        # Règles DATE
        date_rules = data.get('date_regex', {})
        for category, patterns in date_rules.items():
            compiled_patterns = []
            for pattern in patterns:
                try:
                    compiled_patterns.append(re.compile(pattern))
                except re.error as e:
                    logging.warning(f"⚠️ Pattern invalide '{pattern}': {e}")
            
            if compiled_patterns:
                self.date_patterns[category] = compiled_patterns
        
        # Règles PHONE
        phone_rules = data.get('phone_regex', {})
        for category, patterns in phone_rules.items():
            compiled_patterns = []
            for pattern in patterns:
                try:
                    compiled_patterns.append(re.compile(pattern))
                except re.error as e:
                    logging.warning(f"⚠️ Pattern invalide '{pattern}': {e}")
            
            if compiled_patterns:
                self.phone_patterns[category] = compiled_patterns
        
        # Règles NIR
        nir_rules = data.get('nir_regex', {})
        for category, patterns in nir_rules.items():
            compiled_patterns = []
            for pattern in patterns:
                try:
                    compiled_patterns.append(re.compile(pattern))
                except re.error as e:
                    logging.warning(f"⚠️ Pattern invalide '{pattern}': {e}")
            
            if compiled_patterns:
                self.nir_patterns[category] = compiled_patterns
        
        # Règles TIME
        time_rules = data.get('time_regex', {})
        for category, patterns in time_rules.items():
            compiled_patterns = []
            for pattern in patterns:
                try:
                    compiled_patterns.append(re.compile(pattern))
                except re.error as e:
                    logging.warning(f"⚠️ Pattern invalide '{pattern}': {e}")
            
            if compiled_patterns:
                self.time_patterns[category] = compiled_patterns
        
        # Règles EMAIL
        email_rules = data.get('email_regex', {})
        for category, patterns in email_rules.items():
            compiled_patterns = []
            for pattern in patterns:
                try:
                    compiled_patterns.append(re.compile(pattern, re.IGNORECASE))
                except re.error as e:
                    logging.warning(f"⚠️ Pattern invalide '{pattern}': {e}")
            
            if compiled_patterns:
                self.email_patterns[category] = compiled_patterns
        
        # NOTE: Professions supprimées - ne pas pseudonymiser les professions (risque de ré-identification)
        
        total_patterns = (len(self.org_patterns) + len(self.etab_patterns) + len(self.date_patterns) + 
                         len(self.phone_patterns) + len(self.nir_patterns) + len(self.time_patterns) + 
                         len(self.email_patterns))
        logging.info(f"✅ Règles chargées: {len(self.org_patterns)} ORG, {len(self.etab_patterns)} ETAB, "
                    f"{len(self.date_patterns)} DATE, {len(self.phone_patterns)} PHONE, "
                    f"{len(self.nir_patterns)} NIR, {len(self.time_patterns)} TIME, "
                    f"{len(self.email_patterns)} EMAIL")
    
    def detect_entities(self, text: str) -> List[Dict]:
        """Détecte les entités selon les règles avec gestion des entités composées"""
        entities = []
        
        # Patterns de localisation pour détecter les entités composées
        loc_patterns = [
            r"\b(à|au|aux|dans|sur|de|du|des)\s+[A-ZÉÈÀÂÎÙÔÇ][A-Za-zÀ-ÖØ-öø-ÿ''\-\s]+\b",
            r"\bde\s+[A-ZÉÈÀÂÎÙÔÇ][A-Za-zÀ-ÖØ-öø-ÿ''\-\s]+\b"
        ]
        
        # Détecter les ORG avec extension géographique
        for category, patterns in self.org_patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    start, end = match.start(), match.end()
                    base_text = match.group().strip()  # Trim le texte de base
                    
                    if not base_text:  # Ignorer si vide après trim
                        continue
                    
                    # Ajuster les positions après trim
                    left_strip = len(match.group()) - len(match.group().lstrip())
                    start = start + left_strip
                    end = start + len(base_text)
                    
                    # Chercher une extension géographique après l'entité
                    extended_text = base_text
                    extended_end = end
                    
                    # Chercher "de/du/des + lieu" après l'entité de base
                    remaining_text = text[end:]
                    for loc_pattern in loc_patterns:
                        loc_match = re.match(r'\s*' + loc_pattern, remaining_text)
                        if loc_match:
                            extension = loc_match.group().strip()
                            extended_text = base_text + " " + extension
                            extended_end = end + loc_match.end()
                            break
                    
                    entities.append({
                        "start": start,
                        "end": extended_end,
                        "text": extended_text,
                        "label": "ORG",
                        "source": "RULES",
                        "category": category,  # Catégorie spécifique
                        "score": 1.0
                    })
        
        # Détecter les ETAB avec extension géographique
        for category, patterns in self.etab_patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    start, end = match.start(), match.end()
                    base_text = match.group().strip()  # Trim le texte de base
                    
                    if not base_text:  # Ignorer si vide après trim
                        continue
                    
                    # Ajuster les positions après trim
                    left_strip = len(match.group()) - len(match.group().lstrip())
                    start = start + left_strip
                    end = start + len(base_text)
                    
                    # Chercher une extension géographique après l'entité
                    extended_text = base_text
                    extended_end = end
                    
                    # Chercher "de/du/des + lieu" après l'entité de base
                    remaining_text = text[end:]
                    for loc_pattern in loc_patterns:
                        loc_match = re.match(r'\s*' + loc_pattern, remaining_text)
                        if loc_match:
                            extension = loc_match.group().strip()
                            extended_text = base_text + " " + extension
                            extended_end = end + loc_match.end()
                            break
                    
                    entities.append({
                        "start": start,
                        "end": extended_end,
                        "text": extended_text,
                        "label": "ETAB",
                        "source": "RULES",
                        "category": category,  # Catégorie spécifique
                        "score": 1.0
                    })
        
        # Détecter les autres types de données sensibles (sans extension géographique)
        # NOTE: profession_patterns retiré - ne pas pseudonymiser les professions
        sensitive_data_types = [
            ("date_patterns", "DATE"),
            ("phone_patterns", "PHONE"),
            ("nir_patterns", "NIR"),
            ("time_patterns", "TIME"),
            ("email_patterns", "EMAIL")
        ]
        
        for pattern_attr, label in sensitive_data_types:
            patterns_dict = getattr(self, pattern_attr, {})
            for category, patterns in patterns_dict.items():
                for pattern in patterns:
                    for match in pattern.finditer(text):
                        # Trim le texte détecté
                        matched_text = match.group().strip()
                        if not matched_text:  # Ignorer si vide après trim
                            continue
                        
                        # Calculer les vraies positions après trim
                        left_strip = len(match.group()) - len(match.group().lstrip())
                        start_trimmed = match.start() + left_strip
                        end_trimmed = start_trimmed + len(matched_text)
                        
                        entities.append({
                            "start": start_trimmed,
                            "end": end_trimmed,
                            "text": matched_text,
                            "label": label,
                            "source": "RULES",
                            "category": category,
                            "score": 1.0
                        })
        
        logging.info(f"🔧 Règles détectées: {len(entities)} entités")
        return entities


class ChunkedNER:
    """Détecteur NER avec découpage en chunks"""
    
    def __init__(self, model_name: str, max_tokens: int = 400):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.tokenizer = None
        self.pipeline = None
        self._load_model()
    
    def _load_model(self):
        """Charge le modèle NER"""
        logging.info(f"🤖 Chargement modèle NER: {self.model_name}")
        
        try:
            # Essayer d'abord avec AutoTokenizer fast
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        except Exception as e:
            logging.warning(f"⚠️ Tokenizer fast échoué, utilisation CamembertTokenizer: {e}")
            # Fallback vers CamembertTokenizer explicite
            self.tokenizer = CamembertTokenizer.from_pretrained(self.model_name)
        
        model = AutoModelForTokenClassification.from_pretrained(self.model_name)
        self.pipeline = pipeline(
            "token-classification", 
            model=model, 
            tokenizer=self.tokenizer, 
            aggregation_strategy="simple"
        )
        
        logging.info("✅ Modèle NER chargé")
    
    def _split_into_chunks(self, text: str) -> List[Tuple[str, int, int]]:
        """Découpe le texte en chunks avec positions"""
        sentences = text.split('. ')
        chunks = []
        current_pos = 0
        
        i = 0
        while i < len(sentences):
            chunk_sentences = []
            chunk_tokens = 0
            
            while i < len(sentences) and chunk_tokens < self.max_tokens:
                sentence = sentences[i]
                sentence_tokens = len(self.tokenizer.encode(sentence, add_special_tokens=False))
                
                if chunk_tokens + sentence_tokens > self.max_tokens + 50 and chunk_sentences:
                    break
                
                chunk_sentences.append(sentence)
                chunk_tokens += sentence_tokens
                i += 1
            
            if not chunk_sentences:
                chunk_sentences = [sentences[i]]
                i += 1
            
            chunk_text = '. '.join(chunk_sentences)
            if not chunk_text.endswith('.'):
                chunk_text += '.'
            
            chunk_start = current_pos
            chunk_end = chunk_start + len(chunk_text)
            
            chunks.append((chunk_text, chunk_start, chunk_end))
            
            current_pos += len(chunk_text)
            if i < len(sentences):
                current_pos += 1
        
        logging.info(f"📏 Texte divisé en {len(chunks)} chunks")
        return chunks

    def detect_entities(self, text: str) -> List[Dict]:
        """Détecte les entités NER avec découpage en chunks"""
        chunks = self._split_into_chunks(text)
        all_entities = []
        
        for chunk_idx, (chunk_text, chunk_start, chunk_end) in enumerate(chunks):
            logging.debug(f"🧩 Traitement chunk {chunk_idx + 1}/{len(chunks)}")
            
            try:
                chunk_entities = self.pipeline(chunk_text)
                logging.debug(f"🔍 Chunk {chunk_idx} NER résultat brut: {chunk_entities}")
                
                for entity in chunk_entities:
                    label = entity.get("entity_group")
                    if label in {"PER", "ORG", "LOC"}:
                        if entity.get("start") is None or entity.get("end") is None:
                            logging.warning(f"⚠️ Entité avec start/end None: {entity}")
                            continue
                            
                        global_start = chunk_start + int(entity["start"])
                        global_end = chunk_start + int(entity["end"])
                        entity_text = text[global_start:global_end]
                        
                        # 🔧 Trim le texte et ajuster les positions
                        entity_text_trimmed = entity_text.strip()
                        if not entity_text_trimmed:  # Ignorer si vide après trim
                            continue
                        
                        # Calculer le décalage dû au trim
                        left_strip = len(entity_text) - len(entity_text.lstrip())
                        global_start_trimmed = global_start + left_strip
                        global_end_trimmed = global_start_trimmed + len(entity_text_trimmed)
                        
                        all_entities.append({
                            "start": global_start_trimmed,
                            "end": global_end_trimmed,
                            "text": entity_text_trimmed,
                            "label": label,
                            "source": "NER",
                            "score": float(entity.get("score", 0.0)),
                            "chunk": chunk_idx
                        })
                        
            except Exception as ex:
                logging.warning(f"⚠️ Erreur chunk {chunk_idx}: {ex}")
        
        logging.info(f"🤖 NER détecté: {len(all_entities)} entités")
        return all_entities


class ConflictResolver:
    """Résolveur de conflits entre NER, règles et gazetteers avec système de priorité"""
    
    def __init__(self, exclusions: List[str] = None):
        # Liste d'exclusions (termes à ne jamais pseudonymiser)
        self.exclusions = set(exclusions) if exclusions else set()
        
        # Priorités : plus haut = prioritaire (matrice améliorée)
        self.priorities = {
            # Données sensibles critiques
            "RULES_EMAIL": 10.0,
            "RULES_PHONE": 9.5,
            "RULES_NIR": 9.4,
            "ENHANCED_ADDR_FULL": 9.3,
            "RULES_ADDR_FULL": 9.2,
            
            # Fusion TYPE+LOC - priorité TRÈS élevée (TYPE explicite prime sur gazetteer)
            "MERGED_TYPE_LOC_ETAB": 9.1,
            
            # Gazetteers - priorité absolue pour entités connues
            "GAZETTEER_ORG": 9.0,
            "GAZETTEER_ETAB": 9.0,
            
            # Organisations spécialisées (priorité élevée)
            "RULES_ORG_CHU_CH": 8.5,
            "ENHANCED_ORG_CHU_CH": 8.5,
            "RULES_ORG_JUSTICE": 8.4,
            "RULES_ORG_MDPH": 8.3,
            "RULES_ORG_ARS": 8.2,
            "RULES_ORG_DEPARTEMENT": 8.1,
            "RULES_ORG_PREFECTURE": 8.0,
            
            # Sources contextuelles (priorité élevée car très fiables)
            "CONTEXTUAL_PER": 7.9,
            "CONTEXTUAL_PROFESSION": 7.8,
            
            # Établissements spécialisés
            "RULES_EHPAD": 7.9,
            "RULES_IME": 7.8,
            "RULES_ITEP": 7.7,
            "RULES_MECS": 7.6,
            "RULES_ESAT": 7.5,
            "RULES_PROFESSION": 7.0,  # Priorité élevée pour éviter ORG
            
            # Établissements avec noms (ENHANCED)
            "ENHANCED_ETAB_MECS": 6.8,
            "ENHANCED_ETAB_IME": 6.7,
            "ENHANCED_ETAB_ITEP": 6.6,
            "ENHANCED_ETAB_SESSAD": 6.5,
            "ENHANCED_ETAB_ESAT": 6.4,
            "ENHANCED_ETAB_EHPAD": 6.3,
            "ENHANCED_ETAB_MAS": 6.2,
            "ENHANCED_ETAB_FAM": 6.1,
            "ENHANCED_ETAB_FOYER_VIE": 6.0,
            
            # Adresses spécialisées
            "ENHANCED_ADDR_STREET": 5.5,
            "RULES_ADDR_STREET": 5.4,
            "ENHANCED_LOC_CITY": 5.0,
            
            # Temporel
            "RULES_DATE": 4.5,
            "RULES_TIME": 4.4,
            
            # Localisation (priorité plus basse)
            "NER_LOC_CITY": 3.5,
            "NER_LOC": 3.0,
            
            # NER et règles génériques
            "NER_PER": 2.5,
            "RULES_ORG": 2.0,
            "RULES_ETAB": 2.0,
            "NER_ORG": 1.5,
            "DEFAULT": 1.0
        }
        
        # Initialiser les modules d'amélioration si disponibles
        if ENHANCED_MODULES_AVAILABLE:
            self.text_normalizer = TextNormalizer()
            self.stopwords_filter = StopWordsFilter()
            self.priority_matrix = PriorityMatrix()
        else:
            self.text_normalizer = None
            self.stopwords_filter = None
            self.priority_matrix = None
    
    def _calculate_overlap(self, entity1: Dict, entity2: Dict) -> float:
        """Calcule le taux de chevauchement entre deux entités"""
        start1, end1 = entity1["start"], entity1["end"]
        start2, end2 = entity2["start"], entity2["end"]
        
        overlap_start = max(start1, start2)
        overlap_end = min(end1, end2)
        
        if overlap_start >= overlap_end:
            return 0.0
        
        overlap_length = overlap_end - overlap_start
        min_length = min(end1 - start1, end2 - start2)
        
        return overlap_length / min_length if min_length > 0 else 0.0
    
    def _get_priority_key(self, entity: Dict) -> str:
        """Génère la clé de priorité pour une entité"""
        source = entity["source"]
        label = entity["label"]
        category = entity.get("category")
        
        # Fusion TYPE+LOC a une clé spéciale pour priorité très haute
        if source == "MERGED_TYPE_LOC":
            return "MERGED_TYPE_LOC_ETAB"
        elif source == "GAZETTEER":
            return f"GAZETTEER_{label}"
        elif source == "RULES":
            # Utiliser la catégorie spécifique si disponible
            if category:
                return f"RULES_{category}"
            else:
                return f"RULES_{label}"
        else:
            return f"NER_{label}"
    
    def _get_priority(self, entity: Dict) -> float:
        """Retourne la priorité d'une entité"""
        priority_key = self._get_priority_key(entity)
        return self.priorities.get(priority_key, self._get_default_priority(entity))
    
    def _is_valid_entity(self, entity: Dict) -> bool:
        """Filtre les entités invalides"""
        text = entity["text"].strip()
        
        # 🚫 EXCLUSIONS : vérifier si le texte est dans la liste d'exclusion
        if text in self.exclusions or text.lower() in self.exclusions:
            logging.debug(f"🚫 Entité exclue (stoplist): '{text}'")
            return False
        
        # Éliminer les chaînes vides ou trop courtes (sauf codes postaux, téléphones)
        if len(text) < 2:
            logging.debug(f"🗑️ Entité trop courte ignorée: '{text}'")
            return False
        
        # Filtrer les fragments d'adresse isolés
        address_fragments = ["du", "de la", "de l", "des", "Les", "Pôle", "Industrie", "rue de l"]
        if text in address_fragments:
            logging.debug(f"🗑️ Fragment d'adresse ignoré: '{text}'")
            return False
            
        # Éliminer les entités qui sont juste des lettres isolées pour ORG
        if entity["label"] == "ORG" and len(text) <= 2 and text.isalpha():
            logging.debug(f"🗑️ ORG trop court ignoré: '{text}'")
            return False
        
        # Éliminer les abréviations ambiguës sans contexte clair
        ambiguous_short = ["ME", "C", "M", "AS", "ES", "IS"]
        if text.strip() in ambiguous_short and entity["label"] in ["ORG", "ETAB"]:
            logging.debug(f"🗑️ Abréviation ambiguë ignorée: '{text}'")
            return False
            
        # Éliminer les intitulés de fonction/métier 
        job_titles = [
            "technicien intervention sociale et familiale",
            "assistant familial", "éducateur", "psychologue",
            "travailleur social", "aide soignant", "conseiller",
            "référent", "coordinateur", "superviseur"
        ]
        
        if any(job in text.lower() for job in job_titles):
            logging.debug(f"🗑️ Intitulé de fonction ignoré: '{text}'")
            return False
        
        # Éliminer les doublons d'organisations (garder la version la plus spécifique)
        if entity["label"] == "ORG" and entity.get("category") != "ORG_ENTREPRISE_PRIV":
            # Si c'est un nom générique d'organisation déjà couvert par une règle spécialisée
            generic_orgs = ["conseil départemental", "département", "prefecture"]
            if text.lower() in generic_orgs:
                logging.debug(f"🗑️ Organisation générique ignorée (règle spécialisée existe): '{text}'")
                return False
            
        return True
    
    def _enhance_entity_classification(self, entity: Dict, full_text: str) -> Dict:
        """Améliore la classification des entités selon le contexte"""
        text = entity["text"]
        start = entity["start"]
        end = entity["end"]
        
        # Récupérer le contexte élargi (±50 caractères pour être plus précis)
        context_start = max(0, start - 50)
        context_end = min(len(full_text), end + 50)
        context = full_text[context_start:context_end].lower()
        
        # Patterns d'établissements pour expansion PRÉCISE
        etab_patterns = {
            "ETAB_MECS": ["mecs", "maison d'enfants"],
            "ETAB_IME": ["ime"],
            "ETAB_ITEP": ["itep"],
            "ETAB_SESSAD": ["sessad"],
            "ETAB_ESAT": ["esat", "établissement et service d'aide par le travail"],
            "ETAB_EHPAD": ["ehpad", "résidence médicalisée"],
            "ETAB_FOYER_VIE": ["foyer de vie"],
            "ETAB_MAS": ["mas", "maison d'accueil spécialisée"],
            "ETAB_FAM": ["fam", "foyer d'accueil médicalisé"]
        }
        
        # Patterns pour identifier les adresses
        address_patterns = [
            r"\b(?:rue|avenue|boulevard|bd|impasse|place|pl|chemin|allée)\s+",
            r"\b\d{5}\b",  # Code postal
            r"\b\d{1,4}(?:bis|ter)?\s+(?:rue|avenue|boulevard|impasse)",
        ]
        
        # Patterns pour identifier les villes
        city_patterns = [
            r"\b(?:paris|marseille|lyon|toulouse|nice|nantes|montpellier|strasbourg|bordeaux|lille|rennes|reims|le havre|saint-étienne|toulon|grenoble|dijon|angers|nîmes|villeurbanne|saint-denis|le mans|aix-en-provence|clermont-ferrand|brest|limoges|tours|amiens|perpignan|metz|besançon|boulogne-billancourt|orléans|mulhouse|rouen|caen|nancy|saint-paul|argenteuil|montreuil|roubaix|tourcoing|nanterre|avignon|créteil|dunkerque|poitiers|asnières-sur-seine|versailles|courbevoie|vitry-sur-seine|colombes|pau|aulnay-sous-bois|rueil-malmaison|saint-pierre|antibes|saint-maur-des-fossés|cannes|boulogne-sur-mer|nouméa|calais|drancy|cergy|saint-nazaire|colmar|issy-les-moulineaux|noisy-le-grand|évry|villeneuve-d'ascq|la rochelle|antony|troyes|pessac|ivry-sur-seine|clichy|chambéry|lorient|montauban|niort|sète|vincennes|saint-ouen|la seyne-sur-mer|villejuif|saint-andré|clichy-sous-bois|épinay-sur-seine|meaux|merignac|valence|saint-priest|noisy-le-sec|pantin|vénissieux|caluire-et-cuire|bourges|la courneuve|cholet|sartrouville|mantes-la-jolie|bobigny)\b",
            r"\b(?:paris\s+\d{1,2})\b"  # Arrondissements parisiens
        ]
        
        import re
        
        # Classification des adresses et entités spécialisées
        if entity["label"] in ["LOC", "ORG"] and entity["source"] == "NER":
            # D'abord vérifier si c'est un CHU/Centre Hospitalier avant de classer comme ville
            chu_patterns = [
                r"\bCHU\s+(?:de\s+)?[A-ZÉ][\w''\-]+(?:\s+[A-ZÉ][\w''\-]+){0,2}\b",
                r"\b(?:Centre\s+Hospitalier|CHR)\s+(?:de\s+)?[A-ZÉ][\w''\-]+(?:\s+[A-ZÉ][\w''\-]+){0,2}\b"
            ]
            
            for chu_pattern in chu_patterns:
                if re.search(chu_pattern, text, re.IGNORECASE):
                    entity["label"] = "ORG_CHU_CH"
                    entity["category"] = "ORG_CHU_CH"
                    entity["source"] = "ENHANCED"
                    logging.debug(f"🏥 CHU detected: '{text}' → ORG_CHU_CH")
                    return entity
            
            # Types d'établissements qui nécessitent une fusion (à vérifier AVANT reclassification)
            fusion_types = ['EHPAD', 'MAS', 'FAM', 'MECS', 'IME', 'ITEP', 'SESSAD',
                           'ESAT', 'CMPP', 'CAMSP', 'FJT', 'CHRS', 'IMPRO', 'IEM', 
                           'IES', 'SAFEP', 'SSEFS', 'EEAP',
                           'SAVS', 'SAMSAH', 'SSIAD']  # Noms courts pour détection préfixe
            
            # Vérifier si un TYPE d'établissement est immédiatement AVANT (gap ≤ 3 caractères)
            prefix_context = full_text[max(0, start - 15):start].strip().upper()
            has_adjacent_type = any(ftype in prefix_context for ftype in fusion_types)
            
            # Vérifier si c'est une adresse (SAUF si type établissement adjacent)
            is_address = any(re.search(pattern, text.lower()) for pattern in address_patterns)
            if is_address and not has_adjacent_type:
                if re.search(r"\b\d{5}\b", text):
                    entity["label"] = "ADDR_FULL"
                    entity["category"] = "ADDR_FULL"
                else:
                    entity["label"] = "ADDR_STREET"
                    entity["category"] = "ADDR_STREET"
                entity["source"] = "ENHANCED"
                logging.debug(f"🏠 Address detected: '{text}' → {entity['label']}")
                return entity
            
            # Vérifier si c'est une ville (SAUF si type établissement adjacent)
            is_city = any(re.search(pattern, text.lower()) for pattern in city_patterns)
            if is_city and not has_adjacent_type:
                entity["label"] = "LOC_CITY"
                entity["category"] = "LOC_CITY"
                entity["source"] = "ENHANCED"
                logging.debug(f"🌍 City detected: '{text}' → LOC_CITY")
                return entity
            
            # Si type établissement adjacent, préserver LOC pour fusion
            if has_adjacent_type:
                logging.debug(f"🔗 LOC '{text}' préservé pour fusion avec TYPE adjacent (évite reclassification ville/adresse)")
        
        # Détecter si le NER a capturé TYPE+NOM ensemble (ex: "SESSAD Arc-en-Ciel")
        if entity["label"] == "LOC" and entity["source"] == "NER":
            # Types d'établissements à détecter dans le texte NER
            etab_type_patterns = {
                'ETAB_SESSAD': ['SESSAD'],
                'ETAB_EHPAD': ['EHPAD'],
                'ETAB_IME': ['IME'],
                'ETAB_MECS': ['MECS'],
                'ETAB_CHRS': ['CHRS'],
                'ETAB_CMPP': ['CMPP'],
                'ETAB_SAVS': ['SAVS'],
                'ETAB_SAMSAH': ['SAMSAH'],
                'ETAB_MAS': ['MAS'],
                'ETAB_FAM': ['FAM'],
                'ETAB_ITEP': ['ITEP'],
                'ETAB_ESAT': ['ESAT'],
            }
            
            # Vérifier si le texte commence par un type d'établissement
            text_upper = text.upper()
            for etab_category, keywords in etab_type_patterns.items():
                if any(text_upper.startswith(kw) for kw in keywords):
                    logging.debug(f"🏢 NER TYPE+NOM: '{text}' reclassé comme {etab_category}")
                    entity["label"] = etab_category
                    entity["category"] = etab_category
                    entity["source"] = "NER_ENHANCED"
                    return entity
        
        # Expansion d'établissement UNIQUEMENT si l'ancre est présente dans le contexte proche
        # ET que le TYPE d'établissement n'est PAS déjà adjacent (fusion gérée séparément)
        if entity["label"] == "LOC" and entity["source"] == "NER":
            # has_adjacent_type déjà calculé ci-dessus
            
            if not has_adjacent_type:
                for etab_type, keywords in etab_patterns.items():
                    # Vérifier si un mot-clé d'établissement est dans le contexte ET proche (±30 caractères)
                    close_context = full_text[max(0, start - 30):min(len(full_text), end + 30)].lower()
                    if any(keyword in close_context for keyword in keywords):
                        # Heuristiques d'établissement améliorées
                        if _is_establishment_name(text, close_context):
                            logging.debug(f"🏢 LOC→ETAB: '{text}' reclassé comme {etab_type}")
                            entity["label"] = etab_type
                            entity["category"] = etab_type
                            entity["source"] = "ENHANCED"
                            break
            
            # Note: Heuristique ETAB_GENERIC supprimée - chaque type d'établissement
            # doit avoir son pattern spécifique dans rules.yaml

        # Reclassifier les personnes mal étiquetées comme ORG
        if entity["label"] == "ORG" and entity["source"] == "NER":
            # Patterns de noms de personnes
            name_patterns = [
                r"\b[A-Z][a-z]+ [A-Z][A-Z]+\b",  # Prénom NOM
                r"\bM\. [A-Z][A-Z]+\b",          # M. NOM
                r"\bMme [A-Z][A-Z]+\b",          # Mme NOM
                r"\bDr\. [A-Z][A-Z]+\b"         # Dr. NOM
            ]
            
            for pattern in name_patterns:
                if re.search(pattern, text):
                    logging.debug(f"👤 ORG→PER: '{text}' reclassé comme personne")
                    entity["label"] = "PER"
                    entity["source"] = "ENHANCED"
                    break

        # Note: Disambiguation PER vs ETAB supprimée - les noms de personnes
        # détectés par NER restent comme PER (plus de reclassification en ETAB_GENERIC)
        
        # Extension CHU : si une entité contient "CHU" et un nom de ville, étendre pour capturer le nom complet
        if entity["label"] in ["ORG", "LOC"] and entity["source"] == "NER":
            # Chercher si l'entité actuelle contient "CHU"
            if "chu" in text.lower():
                # Examiner le contexte après l'entité pour une éventuelle ville
                context_after = full_text[end:min(len(full_text), end + 50)]
                
                # Chercher une ville immédiatement après (avec espaces possibles)
                import re
                city_match = re.search(r'^\s+([A-Z][a-z]+(?:-[A-Z][a-z]+)*)', context_after)
                if city_match:
                    city_name = city_match.group(1)
                    # Vérifier que ce n'est pas un mot courant qui suivrait CHU
                    excluded_words = ["de", "du", "des", "le", "la", "les", "et", "ou", "avec", "pour"]
                    if city_name.lower() not in excluded_words:
                        # Étendre l'entité pour inclure la ville
                        extended_text = text + " " + city_name
                        entity["text"] = extended_text
                        entity["end"] = end + len(city_match.group(0))
                        entity["label"] = "ORG_CHU_CH"
                        entity["category"] = "ORG_CHU_CH"
                        entity["source"] = "ENHANCED"
                        logging.debug(f"🏥 CHU extended: '{text}' → '{extended_text}' (ORG_CHU_CH)")
                        return entity
        
        return entity

    def _get_default_priority(self, entity: Dict) -> float:
        """Retourne une priorité par défaut pour les catégories non définies"""
        source = entity["source"]
        label = entity["label"]
        
        if source == "ENHANCED":
            return 4.5  # Priorité élevée pour les améliorations
        elif source == "GAZETTEER":
            return 5.0
        elif source == "RULES":
            if label in ["EMAIL", "NIR", "PHONE", "DATE", "TIME"]:
                return 4.0
            elif label == "ETAB":
                return 3.5  # Priorité moyenne pour ETAB non spécifique
            elif label == "ORG":
                return 3.0  # Priorité pour ORG non spécifique
            else:
                return 3.0
        else:  # NER
            if label == "PER":
                return 2.0
            else:
                return 1.0

    def _apply_enhanced_filtering(self, text, entities):
        """Application du filtrage avancé avec les nouvelles classes"""
        if not ENHANCED_MODULES_AVAILABLE:
            return entities
        
        # 1. Filtrage par stopwords
        filtered_entities = []
        for entity in entities:
            entity_text = text[entity['start']:entity['end']]
            if not self.stopwords_filter.is_loc_stopword(entity_text):
                filtered_entities.append(entity)
            else:
                logging.debug(f"🚫 Entité filtrée (stopword): '{entity_text}'")
        
        # 2. Mise à jour des priorités si la matrice est disponible
        for entity in filtered_entities:
            text_content = text[entity['start']:entity['end']]
            entity_type = entity.get('source', '')
            enhanced_priority = self.priority_matrix.get_priority(entity_type)
            if enhanced_priority is not None:
                entity['priority'] = enhanced_priority
                logging.debug(f"🔄 Priorité mise à jour: '{text_content}' → {enhanced_priority}")
        
        return filtered_entities

    def _contextual_disambiguation(self, full_text, entities):
        """Désambiguïsation contextuelle pour corriger les classifications évidentes"""
        if not entities:
            return entities
        
        corrected_entities = []
        corrections_applied = 0
        
        for entity in entities:
            original_entity = dict(entity)  # Copie pour éviter les modifications
            entity_text = entity.get('text', '')
            original_label = entity.get('label', '')
            source = entity.get('source', '')
            
            # Debug: Log toutes les entités ETAB_GENERIC trouvées
            if (original_label.startswith('ETAB_GENERIC') and 
                (source.startswith('NER') or source == 'ENHANCED')):  # Inclure les entités reclassées par enhance
                logging.debug(f"🔍 Entité ETAB_GENERIC détectée: '{entity_text}' source={source} label={original_label}")
                
                # Vérifier si c'est un pattern de personne
                if self._is_person_pattern(entity_text):
                    logging.debug(f"   ✓ Pattern personne détecté pour '{entity_text}'")
                    
                    # Analyse du contexte autour de l'entité (±50 caractères)
                    context_start = max(0, entity['start'] - 50)
                    context_end = min(len(full_text), entity['end'] + 50)
                    context = full_text[context_start:context_end].lower()
                    
                    # Vérifier le contexte
                    if self._has_person_context(context, entity_text):
                        logging.debug(f"   ✓ Contexte personne confirmé pour '{entity_text}'")
                        original_entity['label'] = 'PER'
                        original_entity['source'] = 'CONTEXTUAL_PER'
                        original_entity['category'] = 'PER'
                        corrections_applied += 1
                        logging.info(f"🔄 CORRECTION: '{entity_text}' {original_label} → PER (contexte)")
                    else:
                        logging.debug(f"   ✗ Pas de contexte personne pour '{entity_text}'")
                        logging.debug(f"   Contexte analysé: '{context[:100]}...'")
                else:
                    logging.debug(f"   ✗ Pas un pattern personne: '{entity_text}'")
            
            # Correction PER → PROFESSION pour les titres
            elif (original_label == 'PER' or source.startswith('NER_PER')):
                context_start = max(0, entity['start'] - 50)
                context_end = min(len(full_text), entity['end'] + 50)
                context = full_text[context_start:context_end].lower()
                
                if self._has_professional_title(context, entity_text):
                    logging.debug(f"🔄 Correction PER→PROFESSION: '{entity_text}' (titre détecté)")
                    original_entity['label'] = 'PROFESSION'
                    original_entity['source'] = 'CONTEXTUAL_PROFESSION'
                    original_entity['category'] = 'PROFESSION'
                    corrections_applied += 1
            
            corrected_entities.append(original_entity)
        
        if corrections_applied > 0:
            logging.info(f"🎯 Désambiguïsation contextuelle: {corrections_applied} corrections appliquées")
        
        return corrected_entities
    
    def _is_person_pattern(self, text):
        """Détecte si le texte suit un pattern de nom de personne"""
        import re
        
        # Nettoyer le texte (supprimer espaces et caractères parasites)
        clean_text = text.strip()
        
        # Patterns de noms de personnes avec support des accents
        patterns = [
            r'^[A-ZÀ-ÿ][a-zà-ÿ]+\s+[A-ZÀ-ÿ]{2,}$',             # Prénom NOM (toutes majuscules)
            r'^[A-ZÀ-ÿ][a-zà-ÿ]+\s+[A-ZÀ-ÿ][a-zà-ÿ]+$',       # Prénom Nom (style standard)
            r'^[A-ZÀ-ÿ]\.\s*[A-ZÀ-ÿ][a-zà-ÿ]+$',               # P. Nom
            r'^[A-ZÀ-ÿ][a-zà-ÿ]+\s+[A-ZÀ-ÿ][a-zà-ÿ]+(?:\s+[A-ZÀ-ÿ][a-zà-ÿ]+)?$',  # Prénom Nom MiddleName
        ]
        
        for pattern in patterns:
            if re.match(pattern, clean_text):
                return True
                
        return False
    
    def _has_person_context(self, context, entity_text):
        """Vérifie si le contexte indique qu'il s'agit d'une personne"""
        entity_lower = entity_text.strip().lower()
        
        # Civilités directes
        person_indicators = [
            f'mme {entity_lower}', f'm. {entity_lower}',
            f'monsieur {entity_lower}', f'madame {entity_lower}',
            f'dr {entity_lower}', f'professeur {entity_lower}',
        ]
        
        # Fonctions après le nom
        function_indicators = [
            f'{entity_lower}, directeur', f'{entity_lower}, directrice',
            f'{entity_lower}, responsable', f'{entity_lower}, chef',
            f'{entity_lower}, tutrice', f'{entity_lower}, tuteur',
        ]
        
        # Âge et caractéristiques personnelles (patterns plus flexibles)
        age_indicators = [
            ', ans', ' ans,', f'{entity_lower}, 2', f'{entity_lower}, 3',
            f'{entity_lower}, 4', f'{entity_lower}, 5', f'{entity_lower}, 6'
        ]
        
        # Vérifier tous les indicateurs
        all_indicators = person_indicators + function_indicators + age_indicators
        
        for indicator in all_indicators:
            if indicator in context:
                logging.debug(f"🔍 Indicateur personne trouvé: '{indicator}' pour '{entity_text}'")
                return True
        
        # Debug: afficher le contexte si aucun indicateur trouvé
        logging.debug(f"🔍 Pas d'indicateur personne pour '{entity_text}' dans: '{context[:100]}...'")
        return False
    
    def _has_professional_title(self, context, entity_text):
        """Vérifie si le contexte indique une profession"""
        # Titres professionnels avant le nom
        titles = ['dr ', 'docteur ', 'professeur ', 'pr ']
        
        for title in titles:
            if f'{title}{entity_text.lower()}' in context:
                return True
        
        return False
    
    def _merge_type_location(self, entities: List[Dict], full_text: str) -> List[Dict]:
        """
        Fusionne les TYPES d'établissements (EHPAD, MAS, IME, etc.) avec les LOC/PER adjacents
        détectés par le NER pour créer des entités ETAB complètes.
        
        Logique:
        - Rules détecte "EHPAD" seul
        - NER détecte "Sainte-Gertrude" en LOC
        - Si adjacents (≤ 2 chars d'écart) → fusionner en ETAB_EHPAD "EHPAD Sainte-Gertrude"
        - Si séparés → ignorer le TYPE seul (filtré par exclusions)
        """
        logging.info(f"🔍 _merge_type_location appelée avec {len(entities)} entités")
        
        # Log des entités entrantes pour debug
        for i, e in enumerate(entities[:10]):  # Limiter à 10 pour éviter spam
            logging.debug(f"  [{i}] {e.get('source', 'UNK'):15s} {e.get('label', 'UNK'):8s} {e.get('category', 'N/A'):20s} '{e.get('text', 'N/A')}' (start={e.get('start', 'N/A')}, end={e.get('end', 'N/A')})")
        
        # Types d'établissements à fusionner avec noms propres
        ETAB_TYPES = {
            'EHPAD', 'MAS', 'FAM', 'MECS', 'IME', 'ITEP', 'SESSAD',
            'ESAT', 'CMPP', 'CAMSP', 'FJT', 'CHRS', 'IMPRO', 'IEM', 
            'IES', 'SAFEP', 'SSEFS', 'EEAP',
            'SERVICE_SAVS', 'SERVICE_SAMSAH', 'SERVICE_SSIAD'  # Services avec noms propres (ex: SAVS "Les Passerelles")
        }
        
        # Services NON pseudonymisables (génériques sans nom propre)
        SERVICE_TYPES = {
            'SERVICE_SPASAD', 'SERVICE_SAAD', 'SERVICE_CMP',
            'SERVICE_CATTP', 'SERVICE_CSAPA', 'SERVICE_CAARUD', 'SERVICE_PASS'
        }
        
        merged = []
        skip_indices = set()
        
        # Trier par position pour traitement séquentiel
        sorted_entities = sorted(enumerate(entities), key=lambda x: x[1]['start'])
        
        for i, (idx, entity) in enumerate(sorted_entities):
            if idx in skip_indices:
                continue
            
            # Vérifier si c'est un TYPE d'établissement détecté par RULES
            is_etab_type = (
                entity['source'] == 'RULES' and 
                entity.get('category') in ETAB_TYPES
            )
            
            if is_etab_type and i + 1 < len(sorted_entities):
                # Récupérer l'entité suivante
                next_idx, next_entity = sorted_entities[i + 1]
                
                # Calculer l'écart entre les deux entités
                gap = next_entity['start'] - entity['end']
                
                # Vérifier si l'entité suivante est un LOC ou PER du NER et est adjacente
                is_adjacent_name = (
                    next_entity['label'] in {'LOC', 'PER'} and
                    next_entity['source'] == 'NER' and
                    gap <= 6  # Max 6 chars: guillemets normalisés " ", espaces, apostrophes
                )
                
                # Log pour debug
                if entity.get('category') == 'SESSAD':
                    logging.info(f"🔍 SESSAD: '{entity['text']}' end={entity['end']}, next='{next_entity.get('text', 'N/A')}' start={next_entity.get('start', 'N/A')}, gap={gap}, is_adjacent={is_adjacent_name}")
                
                if is_adjacent_name:
                    # FUSION !
                    type_category = entity.get('category')
                    type_text = entity['text']
                    name_text = next_entity['text']
                    
                    # Construire le texte complet de l'entité fusionnée
                    merged_start = entity['start']
                    merged_end = next_entity['end']
                    merged_text = full_text[merged_start:merged_end]
                    
                    # Log de fusion pour debug
                    logging.info(f"🔗 Fusion: '{type_text}' (start={merged_start}, end={entity['end']}) + '{name_text}' (start={next_entity['start']}, end={merged_end}) → '{merged_text}'")
                    
                    # Catégorie finale : ETAB_<TYPE> (ne pas doubler si déjà ETAB_)
                    final_category = type_category if type_category.startswith('ETAB_') else f"ETAB_{type_category}"
                    
                    merged_entity = {
                        'text': merged_text,
                        'start': merged_start,
                        'end': merged_end,
                        'label': 'ETAB',
                        'category': final_category,
                        'source': 'MERGED_TYPE_LOC',
                        'score': max(entity.get('score', 1.0), next_entity.get('score', 0.8))
                    }
                    
                    merged.append(merged_entity)
                    skip_indices.add(next_idx)  # Ignorer l'entité suivante (déjà fusionnée)
                    
                    logging.debug(f"🔗 Fusion TYPE+LOC: '{type_text}' + '{name_text}' → '{merged_text}' ({final_category})")
                    continue
                    
            # Pas de fusion possible : garder l'entité originale SAUF si c'est un TYPE seul
            if entity['source'] == 'RULES' and entity.get('category') in ETAB_TYPES:
                # TYPE seul sans nom adjacent → ne pas pseudonymiser (filtrage implicite)
                logging.debug(f"🚫 TYPE seul ignoré (pas de nom adjacent): '{entity['text']}' ({entity.get('category')})")
                continue
            
            # Services génériques : conserver mais NE PAS pseudonymiser
            if entity.get('category') in SERVICE_TYPES:
                logging.debug(f"ℹ️ Service générique détecté (non pseudonymisé): '{entity['text']}' ({entity.get('category')})")
                # On ne l'ajoute PAS à merged pour éviter la pseudonymisation
                continue
            
            # Entité valide : ajouter
            merged.append(entity)
        
        logging.info(f"🔗 Fusion TYPE+LOC: {len(entities)} → {len(merged)} entités ({len(entities) - len(merged)} filtrées)")
        return merged
    
    def resolve_conflicts(self, ner_entities: List[Dict], rules_entities: List[Dict], gazetteer_entities: List[Dict] = None, full_text: str = "") -> List[Dict]:
        """Résout les conflits entre NER, règles et gazetteers avec filtrage avancé et désambiguïsation"""
        all_entities = ner_entities + rules_entities
        if gazetteer_entities:
            all_entities.extend(gazetteer_entities)
        
        # Étape 1 : Appliquer le filtrage avancé si disponible  
        if ENHANCED_MODULES_AVAILABLE and full_text:
            all_entities = self._apply_enhanced_filtering(full_text, all_entities)
        
        # Étape 2 : Filtrer les entités invalides et améliorer la classification
        valid_entities = []
        for entity in all_entities:
            if self._is_valid_entity(entity):
                # Améliorer la classification
                enhanced_entity = self._enhance_entity_classification(entity, full_text)
                valid_entities.append(enhanced_entity)
        
        # Étape 3 : Désambiguïsation contextuelle (APRÈS l'amélioration pour corriger les erreurs)
        if full_text:
            valid_entities = self._contextual_disambiguation(full_text, valid_entities)
        
        # Étape 4 : Fusion TYPE + LOC/PER pour établissements (NOUVELLE LOGIQUE)
        if full_text:
            valid_entities = self._merge_type_location(valid_entities, full_text)
        
        resolved = []
        
        # Trier par position
        valid_entities.sort(key=lambda x: x["start"])
        
        for current in valid_entities:
            # Vérifier les conflits avec les entités déjà résolues
            conflicts = []
            for resolved_entity in resolved:
                overlap = self._calculate_overlap(current, resolved_entity)
                if overlap > 0.3:  # Seuil de conflit
                    conflicts.append(resolved_entity)
            
            if not conflicts:
                # Pas de conflit, ajouter directement
                resolved.append(current)
            else:
                # Résoudre le conflit : LONGEST-SPAN-WINS d'abord, puis priorité
                current_length = current['end'] - current['start']
                current_priority = self._get_priority(current)
                
                should_replace = True
                for conflict in conflicts:
                    conflict_length = conflict['end'] - conflict['start']
                    conflict_priority = self._get_priority(conflict)
                    
                    # Règle 1 : Longest-span-wins (si différence > 3 chars)
                    if abs(current_length - conflict_length) > 3:
                        if conflict_length > current_length:
                            should_replace = False
                            break
                    # Règle 2 : Si longueurs similaires, utiliser la priorité
                    elif conflict_priority >= current_priority:
                        should_replace = False
                        break
                
                if should_replace:
                    # Retirer les entités conflictuelles moins prioritaires/courtes
                    for conflict in conflicts:
                        resolved.remove(conflict)
                    resolved.append(current)
                    
                    logging.debug(f"🔄 Conflit résolu: '{current['text']}' ({self._get_priority_key(current)}, len={current_length}) remplace {len(conflicts)} entité(s)")
        
        # Statistiques
        stats = {}
        for entity in resolved:
            key = self._get_priority_key(entity)
            stats[key] = stats.get(key, 0) + 1
        
        logging.info(f"⚖️ Conflits résolus: {len(resolved)} entités finales {stats}")
        
        return resolved
    


def pseudonymize_text(text: str, entities: List[Dict], store: PseudonymStore) -> str:
    """Applique la pseudonymisation sur le texte"""
    entities_sorted = sorted(entities, key=lambda x: x["start"], reverse=True)
    
    result = text
    replacements = 0
    
    for entity in entities_sorted:
        start, end = entity["start"], entity["end"]
        original = entity["text"]
        label = entity["label"]
        category = entity.get("category")  # Récupérer la catégorie spécifique
        
        actual_text = result[start:end]
        if actual_text == original:
            replacement = store.pseudonymize(original, label, category)
            result = result[:start] + replacement + result[end:]
            replacements += 1
            source_info = f"({entity.get('source', 'UNK')})"
            logging.debug(f"🔄 {original} → {replacement} {source_info}")
        else:
            logging.warning(f"⚠️ Mismatch pos {start}-{end}: '{original}' vs '{actual_text}'")
    
    logging.info(f"✅ {replacements} remplacements effectués")
    return result


def main():
    """Point d'entrée principal"""
    parser = argparse.ArgumentParser(description="Pipeline complet NER + Règles + Gazetteers")
    parser.add_argument("--input", required=True, help="Fichier d'entrée")
    parser.add_argument("--output", required=True, help="Fichier de sortie")
    parser.add_argument("--mapping", help="Fichier de mapping")
    parser.add_argument("--load-mapping", help="Fichier de mapping existant à charger")
    parser.add_argument("--rules", default="rules/rules.yaml", help="Fichier de règles")
    parser.add_argument("--gazetteers", default="gazetteer", help="Dossier des gazetteers")
    parser.add_argument("--model", default="Jean-Baptiste/camembert-ner", help="Modèle NER")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING"], default="INFO")
    parser.add_argument("--depseudonymize", action="store_true", help="Mode dépseudonymisation")
    
    args = parser.parse_args()
    
    # Configuration logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )
    
    if args.depseudonymize:
        logging.info("🔓 Mode dépseudonymisation")
        if not args.load_mapping:
            logging.error("❌ Mode dépseudonymisation nécessite --load-mapping")
            sys.exit(1)
    else:
        logging.info("🚀 Pipeline complet NER + Règles + Gazetteers")
    
    # Charger le texte
    text = Path(args.input).read_text(encoding="utf-8")
    
    # Supprimer les guillemets typographiques pour améliorer la détection NER
    # (les guillemets confondent le NER qui inclut le type d'établissement dans la détection)
    original_text = text
    text = text.replace('«', '').replace('»', '').replace('"', '').replace('"', '').replace('"', '')
    
    if text != original_text:
        logging.info(f"📝 Guillemets supprimés pour améliorer détection NER")
    logging.info(f"📖 Texte chargé: {len(text)} caractères")
    
    # Normalisation Unicode si les modules avancés sont disponibles
    original_length = len(text)
    if ENHANCED_MODULES_AVAILABLE:
        normalizer = TextNormalizer()
        text = normalizer.normalize_unicode(text)
        if len(text) != original_length:
            logging.info(f"🔄 Normalisation Unicode: {original_length} → {len(text)} caractères")
    
    # Mode dépseudonymisation
    if args.depseudonymize:
        store = PseudonymStore.load(args.load_mapping)
        result = store.depseudonymize(text)
        logging.info("🔓 Dépseudonymisation terminée")
    else:
        # Mode pseudonymisation
        
        # Charger mapping existant si demandé
        if args.load_mapping:
            store = PseudonymStore.load(args.load_mapping)
        else:
            store = PseudonymStore()
        
        # Détection NER
        ner = ChunkedNER(args.model)
        ner_entities = ner.detect_entities(text)
        
        # Détection par règles
        rules = RulesEngine(args.rules)
        rules_entities = rules.detect_entities(text)
        
        # Détection par gazetteers
        gazetteers = GazetteerEngine(args.gazetteers)
        gazetteer_entities = gazetteers.detect_entities(text)
        
        # Charger les exclusions depuis le gazetteer_exclusions
        exclusions = []
        if 'gazetteer_exclusions' in gazetteers.gazetteers:
            exclusions = [entry['name'] for entry in gazetteers.gazetteers['gazetteer_exclusions']]
            logging.info(f"🚫 {len(exclusions)} exclusions chargées")
        
        # Résolution des conflits
        resolver = ConflictResolver(exclusions=exclusions)
        final_entities = resolver.resolve_conflicts(ner_entities, rules_entities, gazetteer_entities, text)
        
        # Pseudonymisation
        result = pseudonymize_text(text, final_entities, store)
    
    # Sauvegarde
    Path(args.output).write_text(result, encoding="utf-8")
    logging.info(f"💾 Résultat sauvé: {args.output}")
    
    if args.mapping and not args.depseudonymize:
        store.save(args.mapping)
    
    # Statistiques finales
    if not args.depseudonymize:
        logging.info(f"📊 Statistiques: {store.counters}")
    logging.info("🎉 Pipeline terminé!")


if __name__ == "__main__":
    main()