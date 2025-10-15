"""Analyser la détection NER sur les noms d'établissements"""
from pseudonymize import ChunkedNER
import json

# Test avec plusieurs exemples d'établissements
test_text = """L'EHPAD Sainte-Gertrude accueille Mme X. 
Elle fréquente le SESSAD Arc-en-Ciel et l'IME Les Amandiers. 
Un EHPAD du quartier et le CMP local assurent le suivi.
Le MECS Clair Matin et la résidence Les Horizons sont mobilisés.
Il est suivi à l'ITEP Pierre-de-Coubertin.
Un foyer de vie du centre-ville."""

print("📊 Analyse NER des établissements\n")
print("Texte analysé:")
print(test_text)
print("\n" + "="*60 + "\n")

ner = ChunkedNER('Jean-Baptiste/camembert-ner')
entities = ner.detect_entities(test_text)

print(f"✅ NER a détecté {len(entities)} entités:\n")
for e in entities:
    print(f"  [{e['label']:8s}] '{e['text']}' (positions {e['start']}-{e['end']})")

print("\n" + "="*60)
print("\n🔍 Établissements attendus:")
print("  - EHPAD Sainte-Gertrude    → devrait être LOC ou PER")
print("  - SESSAD Arc-en-Ciel       → devrait être LOC")
print("  - IME Les Amandiers        → devrait être LOC")
print("  - MECS Clair Matin         → devrait être LOC ou PER")
print("  - résidence Les Horizons   → devrait être LOC")
print("  - ITEP Pierre-de-Coubertin → devrait être LOC ou PER")
print("\n  - EHPAD du quartier        → ne devrait PAS être détecté comme nom propre")
print("  - CMP local                → ne devrait PAS être détecté")
print("  - foyer de vie du centre   → ne devrait PAS être détecté")
