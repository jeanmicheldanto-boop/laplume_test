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
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline


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
        
        for gazetteer_name, entries in self.gazetteers.items():
            for entry in entries:
                name = entry['name']
                category = entry['category']
                
                # Recherche case-insensitive avec correspondance de mots entiers
                pattern = r'\b' + re.escape(name) + r'\b'
                
                for match in re.finditer(pattern, text, re.IGNORECASE):
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
                        "text": match.group(),
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
        
        total_patterns = len(self.org_patterns) + len(self.etab_patterns) + len(self.date_patterns) + len(self.phone_patterns) + len(self.nir_patterns) + len(self.time_patterns) + len(self.email_patterns)
        logging.info(f"✅ Règles chargées: {len(self.org_patterns)} ORG, {len(self.etab_patterns)} ETAB, {len(self.date_patterns)} DATE, {len(self.phone_patterns)} PHONE, {len(self.nir_patterns)} NIR, {len(self.time_patterns)} TIME, {len(self.email_patterns)} EMAIL")
    
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
                    base_text = match.group()
                    
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
                    base_text = match.group()
                    
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
                        entities.append({
                            "start": match.start(),
                            "end": match.end(),
                            "text": match.group(),
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
        
        # Force l'utilisation du tokenizer lent de CamemBERT pour éviter les problèmes
        from transformers import CamembertTokenizer
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
                
                for entity in chunk_entities:
                    label = entity.get("entity_group")
                    if label in {"PER", "ORG", "LOC"}:
                        global_start = chunk_start + int(entity["start"])
                        global_end = chunk_start + int(entity["end"])
                        entity_text = text[global_start:global_end]
                        
                        all_entities.append({
                            "start": global_start,
                            "end": global_end,
                            "text": entity_text,
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
    
    def __init__(self):
        # Priorités : plus haut = prioritaire
        self.priorities = {
            # Gazetteers - priorité absolue
            "GAZETTEER_ORG": 5,
            "GAZETTEER_ETAB": 5,
            
            # Données sensibles - très prioritaires
            "RULES_EMAIL": 4,
            "RULES_NIR": 4,
            "RULES_PHONE": 4,
            "RULES_DATE": 4,
            "RULES_TIME": 4,
            
            # Établissements spécialisés - haute priorité
            "RULES_EHPAD": 3.9,
            "RULES_MAS": 3.9,
            "RULES_FAM": 3.9,
            "RULES_SESSAD": 3.9,
            "RULES_ITEP": 3.9,
            "RULES_IME": 3.9,
            "RULES_ESAT": 3.9,
            "RULES_SAVS": 3.9,
            "RULES_MECS": 3.9,
            
            # Organisations spécialisées - haute priorité
            "RULES_ORG_MDPH": 3.8,
            "RULES_ORG_CAF": 3.8,
            "RULES_ORG_CHU_CH": 3.8,
            "RULES_ORG_ARS": 3.8,
            "RULES_ORG_ASE": 3.8,
            "RULES_ORG_CPAM": 3.8,
            
            # Règles génériques
            "RULES_ORG": 3,
            "RULES_ETAB": 3,
            
            # NER
            "NER_PER": 2,
            "NER_ORG": 1,
            "NER_LOC": 1
        }
    
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
        
        if source == "GAZETTEER":
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
    
    def _get_default_priority(self, entity: Dict) -> float:
        """Retourne une priorité par défaut pour les catégories non définies"""
        source = entity["source"]
        label = entity["label"]
        
        if source == "GAZETTEER":
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
    
    def resolve_conflicts(self, ner_entities: List[Dict], rules_entities: List[Dict], gazetteer_entities: List[Dict] = None) -> List[Dict]:
        """Résout les conflits entre NER, règles et gazetteers"""
        all_entities = ner_entities + rules_entities
        if gazetteer_entities:
            all_entities.extend(gazetteer_entities)
        
        resolved = []
        
        # Trier par position
        all_entities.sort(key=lambda x: x["start"])
        
        for current in all_entities:
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
                # Résoudre le conflit par priorité
                current_priority = self._get_priority(current)
                
                should_replace = True
                for conflict in conflicts:
                    conflict_priority = self._get_priority(conflict)
                    if conflict_priority >= current_priority:
                        should_replace = False
                        break
                
                if should_replace:
                    # Retirer les entités conflictuelles moins prioritaires
                    for conflict in conflicts:
                        resolved.remove(conflict)
                    resolved.append(current)
                    
                    logging.debug(f"🔄 Conflit résolu: '{current['text']}' ({self._get_priority_key(current)}) remplace {len(conflicts)} entité(s)")
        
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
    logging.info(f"📖 Texte chargé: {len(text)} caractères")
    
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
        
        # Résolution des conflits
        resolver = ConflictResolver()
        final_entities = resolver.resolve_conflicts(ner_entities, rules_entities, gazetteer_entities)
        
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