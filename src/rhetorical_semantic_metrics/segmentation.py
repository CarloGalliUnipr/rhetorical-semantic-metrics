import re
import pandas as pd

def clean_abstract_text(text):
    if pd.isna(text):
        return ''
    text = str(text)
    return re.sub(r'\s+', ' ', text).strip()

def segment_abstracts(records, text_col='Abstract', id_col='PMID', title_col='Title', dataset_col='Dataset', min_chars=6, spacy_model='en_core_web_sm'):
    import spacy
    nlp = spacy.load(spacy_model)
    rows = []
    for _, row in records.iterrows():
        text = clean_abstract_text(row.get(text_col, ''))
        doc = nlp(text)
        sid = 0
        for sent in doc.sents:
            s = sent.text.strip()
            if len(s) < min_chars:
                continue
            sid += 1
            rows.append({
                id_col: row.get(id_col),
                'Dataset': row.get(dataset_col),
                'Title': row.get(title_col, ''),
                'sentence_id': sid,
                'sentence_text': s
            })
    return pd.DataFrame(rows)
