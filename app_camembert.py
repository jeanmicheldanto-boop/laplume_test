# Exemple d'intégration spaCy + CamemBERT NER dans Streamlit
import streamlit as st
import spacy
from transformers import CamembertTokenizer, AutoModelForTokenClassification, pipeline
from presidio_analyzer import RecognizerResult

# Chargement du modèle spaCy pour le prétraitement
nlp = spacy.load("fr_core_news_md")

# Chargement du modèle CamemBERT NER
@st.cache_resource
def get_camembert_ner():
    from transformers import AutoTokenizer
    # Utilisation du tokenizer FAST au lieu de CamembertTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Jean-Baptiste/camembert-ner", use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained("Jean-Baptiste/camembert-ner")
    # Configuration exacte selon la documentation du modèle
    nlp_ner = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
    return nlp_ner

camembert_ner = get_camembert_ner()

st.title("Pseudonymisation avancée avec spaCy + CamemBERT NER")

uploaded_file = st.file_uploader("Déposez un fichier Word (.docx)", type=["docx"])

if uploaded_file:
    import io
    from docx import Document
    doc = Document(io.BytesIO(uploaded_file.read()))
    text = "\n".join([para.text for para in doc.paragraphs])
    st.subheader("Texte original")
    st.write(text)

    # Prétraitement linguistique avec spaCy
    doc_spacy = nlp(text)
    sentences = [sent.text for sent in doc_spacy.sents]

    # NER avec CamemBERT sur chaque phrase
    all_entities = []
    offset = 0
    # Entités sensibles à pseudonymiser (exclut MISC)
    sensitive_entities = ["PER", "ORG", "LOC"]
    
    for sent in sentences:
        ner_results = camembert_ner(sent)
        for ent in ner_results:
            # Vérification des positions ET du type d'entité
            if (ent.get("start") is not None and ent.get("end") is not None 
                and ent["entity_group"] in sensitive_entities):
                # Calcul de l'offset de la phrase dans le texte original
                start_offset = text.find(sent, offset)
                if start_offset != -1:
                    all_entities.append(RecognizerResult(
                        ent["entity_group"],
                        ent["start"] + start_offset,
                        ent["end"] + start_offset,
                        ent["score"],
                        ent["word"]
                    ))
        offset += len(sent)

    # Pseudonymisation par alias unique
    alias_counters = {}
    mapping = {}
    anonymized_text = text
    for r in sorted(all_entities, key=lambda x: x.start, reverse=True):
        ent_type = r.entity_type
        if ent_type not in alias_counters:
            alias_counters[ent_type] = 1
        alias = f"<{ent_type}_{alias_counters[ent_type]}>"
        alias_counters[ent_type] += 1
        # Utilisation du texte original extrait du document au lieu de r.word
        original_text = text[r.start:r.end]
        mapping[alias] = original_text
        anonymized_text = anonymized_text[:r.start] + alias + anonymized_text[r.end:]

    st.subheader("Texte pseudonymisé")
    st.write(anonymized_text)

    # Dépseudonymisation
    if st.button("Dépseudonymiser"):
        depseudo_text = anonymized_text
        for alias, original in mapping.items():
            depseudo_text = depseudo_text.replace(alias, original)
        st.subheader("Texte dépseudonymisé")
        st.write(depseudo_text)
