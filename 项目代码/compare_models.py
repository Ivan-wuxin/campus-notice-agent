"""Reproduce the training-set-only hyperparameter comparison."""
import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline

from notice_agent import KEYWORDS, classify
from train_classifier import read_dataset


def main():
    train, _ = read_dataset(Path("notices.jsonl"))
    texts = [row["text"] for row in train]
    labels = [row["label"] for row in train]
    groups = [row["group"] for row in train]
    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)

    def metrics(predicted):
        return {
            "accuracy": accuracy_score(labels, predicted),
            "macro_f1": f1_score(labels, predicted, labels=list(KEYWORDS),
                                 average="macro", zero_division=0),
        }

    scores = {"keywords": metrics([classify(text) for text in texts])}
    for low, high in ((1, 3), (2, 4), (1, 4)):
        for c in (1, 5, 20):
            model = make_pipeline(TfidfVectorizer(analyzer="char", ngram_range=(low, high)),
                                  LogisticRegression(C=c, max_iter=1000))
            predicted = cross_val_predict(model, texts, labels, groups=groups, cv=cv)
            scores[f"char{low}-{high}_C{c}"] = metrics(predicted)
    result = {"training_size": len(train), "folds": 3, "seed": 42, "scores": scores}
    Path("classification_cv.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
