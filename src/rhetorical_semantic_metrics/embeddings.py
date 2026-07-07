import numpy as np

def compute_sentence_embeddings(sentences, model_name='all-MiniLM-L6-v2', normalize=True, batch_size=32):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    embeddings = model.encode(sentences, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=normalize)
    return np.asarray(embeddings)

def serialize_embedding(vec):
    return ' '.join(str(float(x)) for x in vec)

def deserialize_embedding(text):
    return np.fromstring(text, sep=' ')
