import os

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .config import EMBED_MODEL

load_dotenv()
if os.getenv("OPENAI_API_KEY") is None:
    raise ValueError("OPENAI_API_KEY not found in .env")

client = OpenAI()

# As done in class, use the dot product of two vectors to find the cosine 
# of the angle between two vectors. This will give a measure of similarity. 
def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors"""
    a = np.asarray(vec1, dtype=float)
    b = np.asarray(vec2, dtype=float)
    
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return np.nan

    return float(np.dot(a, b) / denom)

# For comparison, use TF-IDF. 
def add_tfidf_similarity(df, theme_text):
    """Build a traditional TF-IDF thematic score from company description."""
    df = df.copy()

    # Create the corpus from company descriptions and theme. 
    company_texts = df["Company Description"].astype(str).tolist()
    corpus = company_texts + [theme_text]

    # Tokenize text, build vocabulary, count term frequency, apply TF-IDF 
    # weighting for unigram and bigrams. 
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
    )

    tfidf_matrix = vectorizer.fit_transform(corpus).toarray()
    # All rows but the last are company vectors. 
    company_matrix = tfidf_matrix[:-1]
    # The last row is the theme vector. 
    theme_vector = tfidf_matrix[-1]

    # Compute cosine similarity between each company vector and the theme vector. 
    similarities = []
    for row in company_matrix:
        similarities.append(cosine_similarity(row, theme_vector))

    # Create similariy column using tf-idf in the dataframe. 
    df["tfidf_similarity"] = similarities
    return df


@retry(
    reraise=True,
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(Exception),
)
def get_embedding(text, model=EMBED_MODEL):
    """Create an embedding vector for one text string."""
    text = text.replace("\n", " ").strip()

    if text == "":
        return []

    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

# As done in class, use embeddings model from OpenAI to create vector
# embeddings for each company.
def add_embeddings(df):
    """Add one embedding vector per company description."""
    df = df.copy()
    embeddings = []

    for i, text in enumerate(df["Company Description"].astype(str), start=1):
        if text.strip() == "":
            embedding = []
        else:
            embedding = get_embedding(text)

        embeddings.append(embedding)
        print(f"Embeddings: processed {i}/{len(df)}")

    df["embedding_vector"] = embeddings
    return df

# As done in class, use cosine similarities to see how close the company
# vector embeddings are to the theme vector embeddings. 
def add_embedding_similarity(df, theme_text):
    """Build an embedding-based thematic score from company summaries."""
    df = df.copy()
    theme_embedding = get_embedding(theme_text)

    similarities = []
    for embedding in df["embedding_vector"]:
        if isinstance(embedding, list) and len(embedding) > 0:
            similarities.append(cosine_similarity(embedding, theme_embedding))
        else:
            similarities.append(np.nan)

    df["embedding_similarity"] = similarities
    return df
