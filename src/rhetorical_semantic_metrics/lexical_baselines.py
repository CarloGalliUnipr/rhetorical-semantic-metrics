import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

STAT_TOKENS = {'p', 'ci', 'confidence', 'interval', 'or', 'rr', 'hr', 'sd', 'se', 'mean', 'median'}

def tokenise(text):
    return re.findall(r"[A-Za-z0-9.%+-]+", str(text).lower())

def type_token_ratio(text):
    toks = tokenise(text)
    return len(set(toks)) / len(toks) if toks else np.nan

def numeral_density(text):
    toks = tokenise(text)
    return sum(any(ch.isdigit() for ch in t) for t in toks) / len(toks) if toks else np.nan

def statistical_token_density(text):
    toks = tokenise(text)
    return sum(t in STAT_TOKENS or re.match(r"p[<=>]", t) for t in toks) / len(toks) if toks else np.nan

def tfidf_within_results(sentence_df, text_col='sentence_text', label_col='label'):
    res = sentence_df[sentence_df[label_col] == 'Results']
    if len(res) < 2:
        return np.nan
    X = TfidfVectorizer().fit_transform(res[text_col].fillna('').tolist())
    sims = cosine_similarity(X)
    iu = np.triu_indices_from(sims, k=1)
    return float(np.mean(sims[iu]))

def tfidf_conclusion_results(sentence_df, text_col='sentence_text', label_col='label'):
    res = ' '.join(sentence_df.loc[sentence_df[label_col] == 'Results', text_col].fillna('').tolist())
    con = ' '.join(sentence_df.loc[sentence_df[label_col] == 'Conclusion', text_col].fillna('').tolist())
    if not res or not con:
        return np.nan
    X = TfidfVectorizer().fit_transform([res, con])
    return float(cosine_similarity(X[0], X[1])[0, 0])
