import gzip
import json
import math
import re
from pathlib import Path

import nltk
import numpy as np
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class Buscador:
    """Recupera documentos completos con TF-IDF o BM25."""

    def __init__(self, doc_ids, doc_titles, doc_texts, stop_words):
        if len(doc_ids) != len(set(doc_ids)):
            raise ValueError('Los IDs del corpus deben ser únicos')
        self.doc_ids = doc_ids
        self.doc_titles = doc_titles
        self.doc_texts = doc_texts
        self.stop_words = set(stop_words)
        self.stemmer = PorterStemmer()

    def preprocess(self, text):
        """Normaliza, tokeniza, elimina stopwords y aplica stemming."""
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        return [
            self.stemmer.stem(tok)
            for tok in word_tokenize(text)
            if tok.isalpha() and tok not in self.stop_words and len(tok) > 2
        ]

    def ajustar_tfidf(self, tokenized_docs):
        preprocessed_docs = [' '.join(tokens) for tokens in tokenized_docs]
        self.tfidf_vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(preprocessed_docs)

    def search_tfidf(self, query, top_k=10):
        """Devuelve pares de ID y similitud coseno con puntaje positivo."""
        q_text = ' '.join(self.preprocess(query))
        q_vec = self.tfidf_vectorizer.transform([q_text])
        similarities = cosine_similarity(q_vec, self.tfidf_matrix).flatten()
        top_idx = np.argsort(similarities)[::-1][:top_k]
        results = []
        for idx in top_idx:
            score = float(similarities[idx])
            if score <= 0:
                break
            results.append((self.doc_ids[idx], score))
        return results

    def ajustar_bm25(self, tokenized_docs, inverted_index):
        self.tokenized_docs = tokenized_docs
        self.inverted_index = inverted_index
        self.doc_lengths = np.array([len(tokens) for tokens in tokenized_docs], dtype=np.float64)
        self.avgdl = float(np.mean(self.doc_lengths))
        N = len(tokenized_docs)
        df = {term: len(docs) for term, docs in inverted_index.items()}
        self.idf = {
            term: math.log((N - n + 0.5) / (n + 0.5) + 1e-10)
            for term, n in df.items()
        }
        self.k1 = 1.5
        self.b = 0.75

    def search_bm25(self, query, top_k=10):
        """Devuelve pares de ID y puntaje BM25 positivo."""
        q_tokens = self.preprocess(query)
        scores = np.zeros(len(self.doc_ids), dtype=np.float64)
        for term in q_tokens:
            if term not in self.inverted_index:
                continue
            idf_term = self.idf[term]
            for doc_idx in self.inverted_index[term]:
                f = self.tokenized_docs[doc_idx].count(term)
                if f == 0:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * self.doc_lengths[doc_idx] / self.avgdl)
                scores[doc_idx] += idf_term * f * (self.k1 + 1) / denom
        top_idx = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_idx:
            score = float(scores[idx])
            if score <= 0:
                break
            results.append((self.doc_ids[idx], score))
        return results


def guardar_indice(buscador, archivo):
    matriz = buscador.tfidf_matrix
    indice = {
        'doc_ids': buscador.doc_ids,
        'doc_titles': buscador.doc_titles,
        'doc_texts': buscador.doc_texts,
        'stop_words': sorted(buscador.stop_words),
        'tfidf': {
            'vocabulary': {term: int(col) for term, col in buscador.tfidf_vectorizer.vocabulary_.items()},
            'idf': buscador.tfidf_vectorizer.idf_.tolist(),
            'shape': matriz.shape,
            'data': matriz.data.tolist(),
            'indices': matriz.indices.tolist(),
            'indptr': matriz.indptr.tolist(),
        },
        'bm25': {
            'tokenized_docs': buscador.tokenized_docs,
            'inverted_index': buscador.inverted_index,
            'doc_lengths': buscador.doc_lengths.tolist(),
            'avgdl': buscador.avgdl,
            'idf': buscador.idf,
            'k1': buscador.k1,
            'b': buscador.b,
        },
    }
    with gzip.open(archivo, 'wt', encoding='utf-8', compresslevel=6) as salida:
        json.dump(indice, salida, ensure_ascii=False, separators=(',', ':'))


def cargar_indice(archivo):
    """Restaura los modelos y el corpus a partir del índice ya calculado."""
    recursos_nltk = str(archivo.parent / 'nltk_data')
    if recursos_nltk not in nltk.data.path:
        nltk.data.path.insert(0, recursos_nltk)
    nltk.data.find('tokenizers/punkt_tab/english/', paths=[recursos_nltk])

    with gzip.open(archivo, 'rt', encoding='utf-8') as entrada:
        indice = json.load(entrada)
    buscador = Buscador(indice['doc_ids'], indice['doc_titles'], indice['doc_texts'], indice['stop_words'])

    tfidf = indice['tfidf']
    buscador.tfidf_vectorizer = TfidfVectorizer(vocabulary=tfidf['vocabulary'])
    buscador.tfidf_vectorizer.idf_ = np.asarray(tfidf['idf'], dtype=np.float64)
    buscador.tfidf_matrix = csr_matrix(
        (tfidf['data'], tfidf['indices'], tfidf['indptr']),
        shape=tuple(tfidf['shape']),
        dtype=np.float64,
    )

    bm25 = indice['bm25']
    buscador.tokenized_docs = bm25['tokenized_docs']
    buscador.inverted_index = bm25['inverted_index']
    buscador.doc_lengths = np.asarray(bm25['doc_lengths'], dtype=np.float64)
    buscador.avgdl = bm25['avgdl']
    buscador.idf = bm25['idf']
    buscador.k1 = bm25['k1']
    buscador.b = bm25['b']
    return buscador
