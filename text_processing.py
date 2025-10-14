#!/usr/bin/env python3
"""
Module de normalisation et pré-traitement pour la pseudonymisation
"""
import re
import unicodedata
from typing import Dict, List, Tuple

class TextNormalizer:
    """Gestionnaire de normalisation Unicode et textuelle"""
    
    def __init__(self):
        # Mappings de normalisation
        self.apostrophe_map = {'\u02BC': "'", '\u2019': "'", '`': "'"}  # ʼ, ', `
        self.quote_map = {'\u00AB': '"', '\u00BB': '"', '\u201C': '"', '\u201D': '"', '\u201E': '"', '\u201A': "'"}  # «, », ", ", „, ‚
        self.dash_map = {'\u2010': '-', '\u2013': '-', '\u2014': '-', '\u2015': '-'}  # ‐, –, —, ―
        self.space_map = {'\u00A0': ' ', '\u2000': ' ', '\u2001': ' ', '\u2002': ' ', 
                         '\u2003': ' ', '\u2004': ' ', '\u2005': ' ', '\u2006': ' ',
                         '\u2007': ' ', '\u2008': ' ', '\u2009': ' ', '\u200A': ' '}
        
        # Patterns pour détecter les artefacts de copier-coller
        self.copy_paste_patterns = [
            re.compile(r'\n{3,}'),  # Multiples retours ligne
            re.compile(r' {3,}'),   # Multiples espaces
            re.compile(r'\t+'),     # Tabulations
        ]
    
    def normalize_unicode(self, text: str) -> str:
        """Normalise le texte selon Unicode NFC"""
        # Normalisation Unicode NFC
        text = unicodedata.normalize('NFC', text)
        
        # Harmoniser les apostrophes
        for old, new in self.apostrophe_map.items():
            text = text.replace(old, new)
        
        # Harmoniser les guillemets
        for old, new in self.quote_map.items():
            text = text.replace(old, new)
        
        # Harmoniser les tirets
        for old, new in self.dash_map.items():
            text = text.replace(old, new)
        
        # Harmoniser les espaces
        for old, new in self.space_map.items():
            text = text.replace(old, new)
        
        return text
    
    def clean_copy_paste_artifacts(self, text: str) -> Tuple[str, Dict[int, int]]:
        """
        Nettoie les artefacts de copier-coller et retourne un mapping des offsets
        """
        cleaned = text
        offset_map = {}  # old_pos -> new_pos
        
        # Compacter les multiples retours ligne
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        
        # Compacter les multiples espaces (mais garder au moins un)
        cleaned = re.sub(r' {2,}', ' ', cleaned)
        
        # Supprimer les tabulations
        cleaned = cleaned.replace('\t', ' ')
        
        # Créer le mapping des offsets (simple pour l'instant)
        # TODO: Implémenter un mapping précis pour les décalages
        
        return cleaned, offset_map
    
    def normalize_for_matching(self, text: str) -> str:
        """Normalise pour le matching (insensible aux accents/casse)"""
        # Supprimer les accents pour le matching
        text = unicodedata.normalize('NFD', text)
        text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
        
        # Minuscules pour matching
        text = text.lower()
        
        return text


class StopWordsFilter:
    """Filtre les mots vides et ambigus"""
    
    def __init__(self):
        # Stop-words pour localisation
        self.loc_stopwords = {
            'avenue', 'rue', 'bd', 'boulevard', 'place', 'pl', 'secteur',
            'jean', 'saint', 'sainte', 'de', 'du', 'des', 'la', 'le', 'les',
            'nord', 'sud', 'est', 'ouest', 'centre', 'ville'
        }
        
        # Mots ambigus pour bailleurs
        self.habitat_ambiguous = {'habitat'}
        
        # Prefixes/suffixes à éviter pour établissements
        self.etab_exclude_tokens = {
            'avenue', 'rue', 'boulevard', 'place', 'secteur',
            'téléphone', 'phone', 'email', 'mail'
        }
    
    def is_loc_stopword(self, token: str) -> bool:
        """Vérifie si un token est un stop-word de localisation"""
        return token.lower() in self.loc_stopwords
    
    def is_habitat_ambiguous(self, text: str) -> bool:
        """Vérifie si 'Habitat' seul (sans qualificatif)"""
        return text.lower().strip() == 'habitat'
    
    def contains_etab_exclude_tokens(self, text: str) -> bool:
        """Vérifie si le texte contient des tokens à exclure pour ETAB"""
        text_lower = text.lower()
        return any(token in text_lower for token in self.etab_exclude_tokens)


class PriorityMatrix:
    """Matrice de priorités pour la résolution de conflits"""
    
    def __init__(self):
        # Priorités (plus haut = plus prioritaire)
        self.priorities = {
            # Données sensibles critiques
            'ADDR_FULL': 10.0,
            'EMAIL': 9.5,
            'PHONE': 9.4,
            'NIR': 9.3,
            
            # Organisations spécialisées
            'ORG_CHU_CH': 8.5,
            'ORG_JUSTICE': 8.4,
            'ORG_MDPH': 8.3,
            'ORG_ARS': 8.2,
            'ORG_DEPARTEMENT': 8.1,
            'ORG_PREFECTURE': 8.0,
            
            # Établissements spécialisés
            'ETAB_IME': 7.5,
            'ETAB_ITEP': 7.4,
            'ETAB_MECS': 7.3,
            'ETAB_EHPAD': 7.2,
            'ETAB_ESAT': 7.1,
            'ETAB_SCOLAIRE': 7.0,
            
            # Adresses et localisation
            'ADDR_STREET': 6.5,
            'PROFESSION': 6.0,
            
            # Temporel
            'TIME': 5.5,
            'DATE': 5.4,
            
            # Localisation générique
            'LOC_CITY': 4.0,
            'LOC': 3.5,
            
            # Par défaut
            'DEFAULT': 1.0
        }
    
    def get_priority(self, entity_type: str) -> float:
        """Retourne la priorité d'un type d'entité"""
        return self.priorities.get(entity_type, self.priorities['DEFAULT'])
    
    def should_override(self, current_type: str, new_type: str) -> bool:
        """Détermine si new_type doit remplacer current_type"""
        return self.get_priority(new_type) > self.get_priority(current_type)


class PassOrchestrator:
    """Orchestrateur des passes de détection"""
    
    def __init__(self):
        # Ordre des passes (du plus spécifique au plus général)
        self.pass_order = [
            'contacts_identifiers',  # EMAIL, PHONE, NIR, TIME, DATE
            'addresses',            # ADDR_FULL, ADDR_STREET
            'organizations',        # Toutes les ORG_*
            'establishments',       # Toutes les ETAB_*
            'professions',          # PROFESSION
            'locations',            # LOC_CITY, LOC
            'generic_patterns'      # Patterns génériques (fallback)
        ]
        
        self.pass_configs = {
            'contacts_identifiers': {
                'categories': ['EMAIL', 'PHONE', 'NIR', 'TIME', 'DATE'],
                'priority': 10
            },
            'addresses': {
                'categories': ['ADDR_FULL', 'ADDR_STREET'],
                'priority': 9
            },
            'organizations': {
                'categories': ['ORG_CHU_CH', 'ORG_JUSTICE', 'ORG_MDPH', 'ORG_ARS'],
                'priority': 8
            },
            'establishments': {
                'categories': ['ETAB_IME', 'ETAB_ITEP', 'ETAB_MECS', 'ETAB_EHPAD'],
                'priority': 7
            },
            'professions': {
                'categories': ['PROFESSION'],
                'priority': 6
            },
            'locations': {
                'categories': ['LOC_CITY', 'LOC'],
                'priority': 4
            },
            'generic_patterns': {
                'categories': ['ETAB_GENERIC'],
                'priority': 2
            }
        }
    
    def get_pass_order(self) -> List[str]:
        """Retourne l'ordre des passes"""
        return self.pass_order
    
    def get_pass_config(self, pass_name: str) -> Dict:
        """Retourne la configuration d'une passe"""
        return self.pass_configs.get(pass_name, {})