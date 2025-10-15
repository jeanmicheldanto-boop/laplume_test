"""Debug de la fusion TYPE+LOC"""
from pseudonymize import ChunkedNER

text = "L'EHPAD Sainte-Gertrude accueille Mme X. Elle fréquente le SESSAD Arc-en-Ciel et l'IME Les Amandiers. Un EHPAD du quartier et le CMP local assurent le suivi. Le MECS Clair Matin est mobilisé."

print("Texte:", text)
print("\n" + "="*60)

ner = ChunkedNER('Jean-Baptiste/camembert-ner')
entities = ner.detect_entities(text)

print(f"\n✅ NER a détecté {len(entities)} entités:\n")
for e in entities:
    print(f"  [{e['label']:8s}] '{e['text']}' (start={e['start']}, end={e['end']})")
