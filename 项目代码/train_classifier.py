import argparse
import json
import re
from pathlib import Path

from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline

from notice_agent import KEYWORDS, MODEL_PATH, classify


def read_dataset(path):
    records = {"train": [], "test": []}
    groups, texts = {}, {}
    with path.open(encoding="utf-8-sig") as source:
        for number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("label") not in KEYWORDS or row.get("split") not in records:
                raise ValueError(f"第 {number} 行类别或划分无效")
            if any(not isinstance(row.get(key), str) or not row[key].strip()
                   for key in ("text", "group", "source")):
                raise ValueError(f"第 {number} 行缺少正文、来源或同源组")
            split = row["split"]
            group = row["group"].strip()
            normalized = re.sub(r"\s+", "", row["text"])
            if groups.setdefault(group, split) != split or texts.setdefault(normalized, split) != split:
                raise ValueError(f"第 {number} 行与另一划分存在同源组或重复正文")
            records[split].append(row)
    for split, rows in records.items():
        missing = KEYWORDS.keys() - {row["label"] for row in rows}
        if missing:
            raise ValueError(f"{split} 缺少类别：{', '.join(sorted(missing))}")
    return records["train"], records["test"]


def main():
    parser = argparse.ArgumentParser(description="训练校园通知分类器并对比关键词基线")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--report", type=Path, default=Path("classification_results.json"))
    args = parser.parse_args()

    train, test = read_dataset(args.dataset)
    model = make_pipeline(TfidfVectorizer(analyzer="char", ngram_range=(2, 4)),
                          LogisticRegression(max_iter=1000))
    model.fit([row["text"] for row in train], [row["label"] for row in train])
    texts = [row["text"] for row in test]
    labels = [row["label"] for row in test]
    predictions = {
        "keywords": [classify(text) for text in texts],
        "tfidf_logistic": model.predict(texts).tolist(),
    }
    result = {"training_size": len(train), "test_size": len(test), "metrics": {}, "cases": []}
    for name, predicted in predictions.items():
        report = classification_report(labels, predicted, labels=list(KEYWORDS),
                                       output_dict=True, zero_division=0)
        report["accuracy"] = sum(a == b for a, b in zip(labels, predicted)) / len(labels)
        result["metrics"][name] = report
        print(f"{name}: accuracy={report['accuracy']:.3f}, macro F1={report['macro avg']['f1-score']:.3f}")
    for index, row in enumerate(test):
        result["cases"].append({
            "group": row["group"], "source": row["source"], "expected": row["label"],
            **{name: predicted[index] for name, predicted in predictions.items()},
        })
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    dump(model, args.model)


if __name__ == "__main__":
    main()
