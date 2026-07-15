import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nlp = spacy.load("en_core_web_sm")

def extract_entities(text: str):
    doc = nlp(text)
    return [ent.text for ent in doc.ents]

def tfidf_similarity(reference: str, generated: str) -> float:
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([reference, generated])
    return cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]

def check_hallucination(reference: str, generated: str, threshold: float = 0.3) -> dict:
    score = tfidf_similarity(reference, generated)
    ref_ents = set(extract_entities(reference))
    gen_ents = set(extract_entities(generated))
    hallucinated_ents = gen_ents - ref_ents
    return {
        "similarity_score": round(score, 3),
        "hallucinated_entities": list(hallucinated_ents),
        "is_grounded": score >= threshold and len(hallucinated_ents) == 0
    }