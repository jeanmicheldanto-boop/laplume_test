import streamlit as st
from presidio_analyzer import AnalyzerEngine, RecognizerResult, EntityRecognizer
from presidio_anonymizer import AnonymizerEngine
from docx import Document
import openai
import io
import spacy


# Chargement du modèle spaCy français
nlp_fr = spacy.load("fr_core_news_md")

# Recognizer personnalisé spaCy FR
class SpacyFrRecognizer(EntityRecognizer):
    def __init__(self):
        # Ajout de GPE (villes/pays), EMAIL, PHONE
        super().__init__(supported_entities=["PERSON", "ORG", "LOC", "GPE", "EMAIL", "PHONE"], name="SpacyFrRecognizer", supported_language="fr")

    def analyze(self, text, entities, nlp_artifacts=None):
        results = []
        doc = nlp_fr(text)
        for ent in doc.ents:
            # Mapping des labels spaCy vers Presidio
            label_map = {"PER": "PERSON", "ORG": "ORG", "LOC": "LOC", "GPE": "GPE"}
            if ent.label_ in label_map:
                entity_type = label_map[ent.label_]
                if entity_type in entities:
                    results.append(RecognizerResult(entity_type, ent.start_char, ent.end_char, 0.85, ent.text))
        # Détection simple d'email et téléphone par regex
        import re
        for match in re.finditer(r"[\w\.-]+@[\w\.-]+", text):
            results.append(RecognizerResult("EMAIL", match.start(), match.end(), 0.95, match.group()))
        for match in re.finditer(r"\b(?:0|\+33)[1-9](?:[ .-]?\d{2}){4}\b", text):
            results.append(RecognizerResult("PHONE", match.start(), match.end(), 0.95, match.group()))
        return results

analyzer = AnalyzerEngine()
analyzer.registry.add_recognizer(SpacyFrRecognizer())
anonymizer = AnonymizerEngine()

st.title("Pseudonymisation et Synthèse de Documents Word")

uploaded_file = st.file_uploader("Déposez un fichier Word (.docx)", type=["docx"])

if uploaded_file:
    # Lecture du fichier Word
    doc = Document(io.BytesIO(uploaded_file.read()))
    text = "\n".join([para.text for para in doc.paragraphs])
    st.subheader("Texte original")
    st.write(text)

    # Pseudonymisation avec spaCy FR
    entities_fr = ["PERSON", "ORG", "LOC"]
    try:
        results = analyzer.analyze(text=text, entities=entities_fr, language="fr", allow_list=["SpacyFrRecognizer"])
    except KeyError:
        results = SpacyFrRecognizer().analyze(text, entities_fr)

    # Pseudonymisation avancée : alias unique par entité
    alias_counters = {"PERSON": 1, "ORG": 1, "LOC": 1, "GPE": 1, "EMAIL": 1, "PHONE": 1}
    mapping = {}
    anonymized_text = text
    # Pour diagnostic, afficher les attributs de chaque RecognizerResult
    st.write("--- Attributs RecognizerResult ---")
    for r in results:
        st.write(vars(r))
    # Pour éviter les remplacements multiples, on traite de droite à gauche
    for r in sorted(results, key=lambda x: x.start, reverse=True):
        ent_type = r.entity_type
        if ent_type in alias_counters:
            alias = f"<{ent_type}_{alias_counters[ent_type]}>"
            alias_counters[ent_type] += 1
        else:
            alias = f"<{ent_type}>"
        # Diagnostic : essayer tous les attributs possibles
        ent_text = getattr(r, 'entity_text', None) or getattr(r, 'text', None) or getattr(r, 'value', None) or getattr(r, '_text', None)
        mapping[alias] = ent_text
        anonymized_text = anonymized_text[:r.start] + alias + anonymized_text[r.end:]

    st.subheader("Texte pseudonymisé")
    st.write(anonymized_text)

    # Options d'appel OpenAI
    option = st.selectbox("Choisissez une option de synthèse", ["Synthèse courte", "Synthèse 1 page", "Qualité rédactionnelle"])
    prompt_map = {
        "Synthèse courte": "Fais une synthèse très courte du texte suivant:",
        "Synthèse 1 page": "Fais une synthèse d'une page du texte suivant:",
        "Qualité rédactionnelle": "Améliore la qualité rédactionnelle du texte suivant:"
    }
    if st.button("Lancer l'API OpenAI"):
        openai.api_key = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else ""
        prompt = f"{prompt_map[option]}\n{anonymized_text}"
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        ai_text = response.choices[0].message.content
        st.subheader("Texte généré par OpenAI (pseudonymisé)")
        st.write(ai_text)

        # Dépseudonymisation : remplacement des alias par les originaux
        depseudo_text = ai_text
        for alias, original in mapping.items():
            depseudo_text = depseudo_text.replace(alias, original)
        st.subheader("Texte final (dépseudonymisé)")
        st.write(depseudo_text)
