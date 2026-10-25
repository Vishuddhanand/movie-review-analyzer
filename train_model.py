#!/usr/bin/env python3
"""
train_model.py
--------------
Trains the sentiment model on the IMDB 50K movie-review dataset and writes:

    models/vectorizer.pkl        TF-IDF vectorizer (1-2 grams)
    models/sentiment_model.pkl   Logistic Regression classifier
    models/metrics.json          accuracy, confusion matrix, model comparison, top words
    models/samples.json          example reviews for the "Try a sample" button

Usage
-----
    python train_model.py                              # uses data/IMDB Dataset.csv
    python train_model.py --data "path/to/IMDB Dataset.csv"
    python train_model.py --sample 5000                # quick run on a subset

The CSV needs a `review` column and a `sentiment` column
(values "positive"/"negative", or 1/0).
"""

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from nlp_utils import clean_text

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
RANDOM_STATE = 42


def load_dataset(path: Path, sample: int | None) -> pd.DataFrame:
    print(f"[data] reading {path}")
    df = pd.read_csv(path, on_bad_lines="skip", engine="python")
    df = df.dropna(subset=["review", "sentiment"]).drop_duplicates(subset=["review"])

    # accept "positive"/"negative" as well as 1/0
    labels = df["sentiment"].astype(str).str.strip().str.lower()
    df["label"] = labels.map({"positive": 1, "negative": 0, "1": 1, "0": 0})
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    if sample and sample < len(df):
        df = df.sample(sample, random_state=RANDOM_STATE)

    df["review"] = df["review"].map(clean_text)
    df = df.reset_index(drop=True)
    print(f"[data] {len(df):,} reviews | positive={int(df.label.sum()):,} negative={int((1 - df.label).sum()):,}")
    return df


def evaluate(y_true, y_pred) -> dict:
    report = classification_report(
        y_true, y_pred, target_names=["negative", "positive"], output_dict=True, zero_division=0
    )
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(report["weighted avg"]["precision"], 4),
        "recall": round(report["weighted avg"]["recall"], 4),
        "f1": round(report["weighted avg"]["f1-score"], 4),
        "per_class": {
            cls: {k: round(report[cls][k], 4) for k in ("precision", "recall", "f1-score")}
            | {"support": int(report[cls]["support"])}
            for cls in ("negative", "positive")
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def timed_fit(name, model, X_train, y_train, X_test, y_test, note=""):
    start = time.time()
    model.fit(X_train, y_train)
    seconds = time.time() - start
    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"[bench] {name:<46} accuracy={accuracy:.4f}  ({seconds:.1f}s)")
    return {"name": name, "accuracy": round(accuracy, 4), "train_seconds": round(seconds, 1), "note": note}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=str(BASE_DIR / "data" / "IMDB Dataset.csv"))
    parser.add_argument("--sample", type=int, default=None, help="train on a random subset (faster)")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(
            f"Dataset not found: {data_path}\n"
            "Download 'IMDB Dataset.csv' (Kaggle: 'IMDB Dataset of 50K Movie Reviews') "
            "and put it in the data/ folder, or pass --data <path>."
        )

    MODELS_DIR.mkdir(exist_ok=True)
    df = load_dataset(data_path, args.sample)

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=RANDOM_STATE, stratify=df["label"]
    )
    X_train_text, X_test_text = train_df["review"], test_df["review"]
    y_train, y_test = train_df["label"].to_numpy(), test_df["label"].to_numpy()
    print(f"[split] train={len(train_df):,} test={len(test_df):,}")

    # ---------------------------------------------------------------- features
    # Stop-words are kept on purpose: words like "not" and "but" carry sentiment.
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), min_df=3, max_df=0.9, max_features=150_000, sublinear_tf=True
    )
    start = time.time()
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)
    print(f"[tfidf] {X_train.shape[1]:,} features ({time.time() - start:.1f}s)")

    # ---------------------------------------------------------- pick C for LR
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=RANDOM_STATE, stratify=y_train
    )
    best_c, best_score = 1.0, 0.0
    for c in (1, 4, 10, 20):
        score = accuracy_score(
            y_val, LogisticRegression(C=c, solver="liblinear").fit(X_fit, y_fit).predict(X_val)
        )
        print(f"[tune] C={c:<3} validation accuracy={score:.4f}")
        if score > best_score:
            best_c, best_score = c, score

    # ------------------------------------------------------------ benchmark
    comparison = []
    final_model = LogisticRegression(C=best_c, solver="liblinear")
    comparison.append(timed_fit(
        f"Logistic Regression (C={best_c})", final_model, X_train, y_train, X_test, y_test,
        "Deployed model: TF-IDF 1-2 grams",
    ))
    comparison.append(timed_fit(
        "Linear SVM", LinearSVC(C=0.5), X_train, y_train, X_test, y_test, "TF-IDF 1-2 grams"
    ))
    comparison.append(timed_fit(
        "Multinomial Naive Bayes", MultinomialNB(alpha=0.1), X_train, y_train, X_test, y_test, "TF-IDF 1-2 grams"
    ))

    # The approach from the lab notebook: 5000 TF-IDF features -> LSA(100) -> Logistic Regression
    nb_vec = TfidfVectorizer(max_features=5000, stop_words="english")
    nb_train = nb_vec.fit_transform(X_train_text)
    nb_test = nb_vec.transform(X_test_text)
    lsa = TruncatedSVD(n_components=100, random_state=RANDOM_STATE)
    lsa_train = lsa.fit_transform(nb_train)
    lsa_test = lsa.transform(nb_test)
    comparison.append(timed_fit(
        "LSA (100 topics) + Logistic Regression", LogisticRegression(max_iter=1000),
        lsa_train, y_train, lsa_test, y_test, "Approach used in the lab notebook",
    ))

    # ----------------------------------------------------------- final report
    metrics = evaluate(y_test, final_model.predict(X_test))
    print(f"\n[final] accuracy={metrics['accuracy']:.4f}  f1={metrics['f1']:.4f}")
    print(f"[final] confusion matrix (rows=true, cols=pred): {metrics['confusion_matrix']}")

    feature_names = vectorizer.get_feature_names_out()
    coef = final_model.coef_[0]
    order = np.argsort(coef)

    def top_terms(indices):
        return [{"term": feature_names[i], "weight": round(float(coef[i]), 3)} for i in indices]

    # 2-D LSA projection of 500 test reviews (the notebook's scatter plot)
    rng = np.random.RandomState(RANDOM_STATE)
    pick = rng.choice(len(y_test), size=min(500, len(y_test)), replace=False)
    scatter = [
        {"x": round(float(lsa_test[i, 0]), 4), "y": round(float(lsa_test[i, 1]), 4), "label": int(y_test[i])}
        for i in pick
    ]

    metrics.update({
        "model_name": f"Logistic Regression on TF-IDF (1-2 grams), C={best_c}",
        "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "dataset": {
            "total": int(len(df)),
            "train": int(len(train_df)),
            "test": int(len(test_df)),
            "positive": int(df.label.sum()),
            "negative": int((1 - df.label).sum()),
            "features": int(X_train.shape[1]),
        },
        "comparison": sorted(comparison, key=lambda r: -r["accuracy"]),
        "top_positive": top_terms(order[::-1][:30]),
        "top_negative": top_terms(order[:30]),
        "lsa_scatter": scatter,
        "lsa_variance": round(float(lsa.explained_variance_ratio_[:2].sum()), 4),
        "environment": {"python": platform.python_version(), "scikit_learn": sklearn.__version__},
    })

    # ------------------------------------------------------------- save files
    joblib.dump(vectorizer, MODELS_DIR / "vectorizer.pkl", compress=3)
    joblib.dump(final_model, MODELS_DIR / "sentiment_model.pkl", compress=3)
    (MODELS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # sample reviews: short, correctly classified, half positive / half negative
    preds = final_model.predict(X_test)
    candidates = test_df.assign(pred=preds, length=test_df["review"].str.len())
    candidates = candidates[(candidates.pred == candidates.label) & candidates.length.between(180, 650)]
    samples = []
    for label, name in ((1, "positive"), (0, "negative")):
        chosen = candidates[candidates.label == label].sample(min(12, (candidates.label == label).sum()), random_state=RANDOM_STATE)
        samples += [{"text": t, "actual": name} for t in chosen["review"]]
    rng.shuffle(samples)
    (MODELS_DIR / "samples.json").write_text(json.dumps(samples, indent=1))

    # example CSV for the "Analyze a CSV" tab: random held-out reviews, NOT filtered for
    # correctness, so the accuracy shown in the app is a real number
    static_dir = BASE_DIR / "static"
    static_dir.mkdir(exist_ok=True)
    example = test_df[test_df["review"].str.len() <= 1200].sample(40, random_state=7)
    example = pd.DataFrame({
        "review": example["review"],
        "sentiment": example["label"].map({1: "positive", 0: "negative"}),
    })
    example.to_csv(static_dir / "example_reviews.csv", index=False)

    print("\n[saved]")
    for f in sorted(MODELS_DIR.iterdir()):
        print(f"  models/{f.name:<22} {f.stat().st_size / 1024:>9,.0f} KB")


if __name__ == "__main__":
    main()
