"""Classical dense retrieval lane: L2-normalized TF-IDF vectors + cosine similarity.

This is a REAL dense-vector retriever (vectors, dot products, norms) using the
classical TF-IDF basis — no downloads, no network, deterministic. It is NOT a
neural embedding model; the estate labels it `tfidf-dense`. A neural adapter
plugs into the same `rank(query, doc_ids)` interface later without touching
the harness.
"""
import math
from collections import Counter

from .bm25 import tokenize


class TfidfDense:
    def __init__(self, docs):
        self.N = len(docs)
        tf = [Counter(tokenize(d)) for d in docs]
        df = Counter()
        for t in tf:
            for term in t:
                df[term] += 1
        self.idf = {term: math.log((1 + self.N) / (1 + n)) + 1.0 for term, n in df.items()}
        self.vectors = []
        for t in tf:
            v = {term: f * self.idf[term] for term, f in t.items()}
            norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
            self.vectors.append({term: x / norm for term, x in v.items()})

    def _query_vector(self, query):
        t = Counter(tokenize(query))
        v = {term: f * self.idf.get(term, math.log((1 + self.N) / 2.0) + 1.0)
             for term, f in t.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {term: x / norm for term, x in v.items()}

    def rank(self, query, doc_ids):
        q = self._query_vector(query)
        scores = []
        for v in self.vectors:
            s = sum(w * v.get(term, 0.0) for term, w in q.items())
            scores.append(s)
        return [doc_ids[i] for i in sorted(range(len(doc_ids)), key=lambda i: -scores[i])]
