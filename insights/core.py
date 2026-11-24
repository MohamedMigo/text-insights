"""Text classification, explanations and semantic search on the 20 Newsgroups corpus."""
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import make_pipeline

CATS = ["sci.space", "rec.autos", "talk.politics.misc", "comp.graphics", "sci.med", "rec.sport.hockey"]
STRIP = ("headers", "footers", "quotes")


def load(subset: str):
    return fetch_20newsgroups(subset=subset, categories=CATS, remove=STRIP, random_state=0)


def train_classifier():
    tr = load("train")
    clf = make_pipeline(TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2, stop_words="english"),
                        LogisticRegression(C=10, max_iter=1000))
    clf.fit(tr.data, tr.target)
    return clf, tr.target_names


def evaluate(clf) -> dict:
    te = load("test")
    p = clf.predict(te.data)
    return {"macro_f1": round(f1_score(te.target, p, average="macro"), 4),
            "report": classification_report(te.target, p, target_names=te.target_names, output_dict=True)}


def explain(clf, text: str, names, top: int = 6) -> dict:
    """Which words pushed the prediction? Contribution = tf-idf weight * class coefficient."""
    vec, lr = clf.steps[0][1], clf.steps[1][1]
    x = vec.transform([text])
    k = int(lr.predict(x)[0])
    contrib = x.multiply(lr.coef_[k]).toarray().ravel()
    terms = vec.get_feature_names_out()
    idx = np.argsort(-contrib)[:top]
    return {"label": names[k], "confidence": round(float(lr.predict_proba(x)[0, k]), 3),
            "top_words": [(terms[i], round(float(contrib[i]), 3)) for i in idx if contrib[i] > 0]}


class SemanticSearch:
    """LSA: TF-IDF followed by SVD, so 'rocket' can match 'orbit' via co-occurrence."""

    def __init__(self, docs: list[str], dims: int = 120):
        self.docs = docs
        self.vec = TfidfVectorizer(sublinear_tf=True, stop_words="english", min_df=2)
        self.svd = TruncatedSVD(dims, random_state=0)
        Z = self.svd.fit_transform(self.vec.fit_transform(docs))
        self.Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)

    def query(self, q: str, k: int = 3):
        z = self.svd.transform(self.vec.transform([q]))
        z = z / (np.linalg.norm(z) + 1e-9)
        sims = self.Z @ z.ravel()
        return [(int(i), round(float(sims[i]), 3), self.docs[i][:120].replace("\n", " ")) for i in np.argsort(-sims)[:k]]
