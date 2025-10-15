from transformers import pipeline

# Texte original avec guillemets
original = 'SESSAD « Arc-en-Ciel » (17 rue des Arts, 59800 Lille)'
# Texte avec guillemets supprimés
cleaned = original.replace('«', '').replace('»', '').replace('"', '').replace('"', '').replace('"', '')

print(f"Texte original: {original}")
print(f"Texte nettoyé:  {cleaned}")
print(f"Longueur: {len(cleaned)}")

ner = pipeline("ner", model="Jean-Baptiste/camembert-ner", aggregation_strategy="simple", device=-1)
entities = ner(cleaned)

print(f"\nNER détecté {len(entities)} entités:")
for e in entities:
    print(f"  {e['entity_group']:5s} '{e['word']}' (start={e['start']}, end={e['end']}, score={e['score']:.3f})")
