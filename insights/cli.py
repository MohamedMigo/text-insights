import json
import sys

from .core import SemanticSearch, evaluate, explain, load, train_classifier

if __name__ == "__main__":
    clf, names = train_classifier()
    print("macro F1:", evaluate(clf)["macro_f1"])
    text = " ".join(sys.argv[1:]) or "The shuttle launch was delayed because of a fuel leak in the rocket booster"
    print(json.dumps(explain(clf, text, names), indent=2))
    ss = SemanticSearch(load("train").data)
    for hit in ss.query(text):
        print(hit)
