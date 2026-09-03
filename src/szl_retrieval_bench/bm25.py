"""BM25 sparse retrieval. Stdlib only. No downloads, no network."""
import math
from collections import Counter


def tokenize(text):
    return [w.strip(".,;:!?()[]\"'").lower() for w in text.split() if w.strip(".,;:!?()[]\"'")]


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(tokenize(d)) for d in docs]
        self.dl = [sum(t.values()) for t in self.tf]
        self.avgdl = sum(self.dl) / max(1, len(self.dl))
        df = Counter()
        for t in self.tf:
            for term in t:
                df[term] += 1
        self.N = len(docs)
        self.idf = {term: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for term, n in df.items()}

    def score(self, query):
        q = tokenize(query)
        out = []
        for i, t in enumerate(self.tf):
            s = 0.0
            for term in q:
                if term not in t:
                    continue
                f = t[term]
                s += self.idf.get(term, 0.0) * (f * (self.k1 + 1)) / (
                    f + self.k1 * (1 - self.b + self.b * self.dl[i] / max(1e-9, self.avgdl)))
            out.append(s)
        return out

    def rank(self, query, doc_ids):
        scores = self.score(query)
        return [doc_ids[i] for i in sorted(range(len(doc_ids)), key=lambda i: -scores[i])]
