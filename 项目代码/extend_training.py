"""Compare additional text models using out-of-fold training predictions."""
import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from train_classifier import read_dataset
from notice_agent import KEYWORDS


def main():
    train, _ = read_dataset(Path("notices.jsonl"))
    y = [row["label"] for row in train]
    groups = [row["group"] for row in train]
    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
    results = {}
    for scope in ("full", "title"):
        texts = [row["text"] if scope == "full" else row["text"].split("。", 1)[0]
                 for row in train]
        for name, classifier in (("svm_c1", LinearSVC(C=1, random_state=42)),
                                 ("svm_c5", LinearSVC(C=5, random_state=42)),
                                 ("complement_nb", ComplementNB(alpha=0.1))):
            model = make_pipeline(TfidfVectorizer(analyzer="char", ngram_range=(1, 3)),
                                  classifier)
            pred = cross_val_predict(model, texts, y, groups=groups, cv=cv)
            results[f"{scope}_{name}"] = {
                "correct": sum(a == b for a, b in zip(y, pred)),
                "accuracy": accuracy_score(y, pred),
                "macro_f1": f1_score(y, pred, labels=list(KEYWORDS),
                                     average="macro", zero_division=0),
            }
    Path("extended_cv.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
