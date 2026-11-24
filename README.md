# Text Insights

Three NLP tools on the public 20 Newsgroups corpus (6 topics: space, autos, politics, graphics, medicine, hockey):

1. **Topic classifier**: TF-IDF (1-2 grams) + Logistic Regression. **Macro F1 = 0.847** on the official test split with headers, footers and quoted replies removed (so it cannot cheat on metadata).
2. **Explainability**: for any text, shows which words pushed the prediction (tf-idf weight x class coefficient).
3. **Semantic search**: LSA (TF-IDF + SVD) so related words match even without exact overlap.

## Run
```bash
pip install -r requirements.txt
python -m insights.cli "The shuttle launch was delayed because of a fuel leak in the rocket booster"
pytest -q
```
The corpus downloads automatically the first time (scikit-learn caches it).

Example output:
```
macro F1: 0.8471
{"label": "sci.space", "confidence": ..., "top_words": [["shuttle launch", 0.158], ["leak", 0.107], ...]}
(3394, 0.796, 'Why use a ground launch pad. ...')
```
