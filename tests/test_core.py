from insights.core import SemanticSearch, evaluate, explain, load, train_classifier


def test_classifier_quality_and_explanation():
    clf, names = train_classifier()
    assert evaluate(clf)["macro_f1"] > 0.8
    r = explain(clf, "The rocket reached orbit after the NASA launch", names)
    assert r["label"] == "sci.space" and r["top_words"]


def test_semantic_search_returns_relevant():
    docs = load("train").data[:1500]
    hits = SemanticSearch(docs).query("hockey goalie playoff", 3)
    assert len(hits) == 3 and hits[0][1] >= hits[-1][1]
