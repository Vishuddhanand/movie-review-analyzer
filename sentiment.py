"""
sentiment.py
------------
Loads the trained .pkl files and turns a review into:
    - a verdict with probabilities
    - word-by-word highlighting (which words pushed the prediction, and which way)
    - the strongest positive / negative phrases
"""

import csv
import io
import json
import re
from pathlib import Path

import joblib
import numpy as np

from nlp_utils import clean_text, MAX_CHARS

MODELS_DIR = Path(__file__).resolve().parent / "models"
_WORD = re.compile(r"\b\w\w+\b")  # same token pattern scikit-learn's TF-IDF uses


class ModelNotFound(RuntimeError):
    pass


class SentimentEngine:
    def __init__(self, models_dir: Path = MODELS_DIR):
        try:
            self.vectorizer = joblib.load(models_dir / "vectorizer.pkl")
            self.model = joblib.load(models_dir / "sentiment_model.pkl")
        except FileNotFoundError as exc:
            raise ModelNotFound(
                "Model files not found in models/. Run:  python train_model.py"
            ) from exc

        self.metrics = self._read_json(models_dir / "metrics.json", {})
        self.samples = self._read_json(models_dir / "samples.json", [])

        self.vocab = self.vectorizer.vocabulary_
        self.terms = self.vectorizer.get_feature_names_out()
        self.coef = self.model.coef_[0]
        # Scale word weights so the strongest ~1% of terms map to full highlight intensity.
        self.scale = float(np.percentile(np.abs(self.coef), 99)) or 1.0

    @staticmethod
    def _read_json(path: Path, default):
        try:
            return json.loads(path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return default

    # ------------------------------------------------------------------ verdict
    @staticmethod
    def _verdict(p_pos: float) -> dict:
        confidence = max(p_pos, 1 - p_pos)
        label = "positive" if p_pos >= 0.5 else "negative"
        if confidence < 0.60:
            summary = "Mixed"
        elif confidence < 0.80:
            summary = "Leans positive" if label == "positive" else "Leans negative"
        else:
            summary = "Positive" if label == "positive" else "Negative"
        return {
            "label": label,
            "summary": summary,
            "positive": round(float(p_pos), 4),
            "negative": round(float(1 - p_pos), 4),
            "confidence": round(float(confidence), 4),
        }

    # --------------------------------------------------------------- highlight
    def _highlight(self, text: str, X) -> list:
        """Split the review into segments, each tagged with a weight in [-1, 1].

        The model's score is a sum of (TF-IDF value x coefficient) over the review's
        features. Here every feature's contribution is shared across the words it covers
        (a two-word phrase gives half to each word), so the shading is an exact
        decomposition of the prediction and negation like "not recommend" shows up correctly.
        """
        matches = list(_WORD.finditer(text))
        lowered = [m.group().lower() for m in matches]

        positions = {}  # feature text -> list of word indexes where it starts
        for i, word in enumerate(lowered):
            positions.setdefault(word, []).append(i)
            if i + 1 < len(lowered):
                positions.setdefault(f"{word} {lowered[i + 1]}", []).append(i)

        weights = np.zeros(len(lowered))
        row = X.tocoo()
        for col, value in zip(row.col, row.data):
            term = self.terms[col]
            starts = positions.get(term)
            if not starts:
                continue
            contribution = float(value * self.coef[col])
            width = term.count(" ") + 1
            share = contribution / (len(starts) * width)
            for start in starts:
                for j in range(start, start + width):
                    weights[j] += share

        # Intensity is relative to the strongest word, with a floor so that a review
        # with only faint signals does not get painted at full strength.
        scale = max(float(np.abs(weights).max()) if len(weights) else 0.0, 0.5)
        weights = np.clip(weights / scale, -1, 1)

        segments, cursor = [], 0

        def push(chunk, weight):
            if not chunk:
                return
            if segments and segments[-1]["w"] == weight == 0:
                segments[-1]["t"] += chunk
            else:
                segments.append({"t": chunk, "w": weight})

        for match, weight in zip(matches, weights):
            push(text[cursor:match.start()], 0)
            push(match.group(), round(float(weight), 2) if abs(weight) >= 0.1 else 0)
            cursor = match.end()
        push(text[cursor:], 0)
        return segments

    def _drivers(self, X, k: int = 6) -> dict:
        """Terms (including 2-word phrases) that contributed most in this review."""
        row = X.tocoo()
        contributions = row.data * self.coef[row.col]
        order = np.argsort(contributions)
        negative = [(self.terms[row.col[i]], contributions[i]) for i in order[:k] if contributions[i] < 0]
        positive = [(self.terms[row.col[i]], contributions[i]) for i in order[::-1][:k] if contributions[i] > 0]
        top = max([abs(c) for _, c in positive + negative] or [1])
        fmt = lambda pairs: [
            {"term": t, "strength": round(abs(float(c)) / top, 3)}
            for t, c in pairs
            if abs(float(c)) / top >= 0.15  # hide terms that barely mattered
        ]
        return {"positive": fmt(positive), "negative": fmt(negative)}

    # ---------------------------------------------------------------- public API
    def predict(self, text: str) -> dict:
        text = clean_text(text)[:MAX_CHARS]
        X = self.vectorizer.transform([text])
        p_pos = float(self.model.predict_proba(X)[0, 1])
        return {
            **self._verdict(p_pos),
            "highlight": self._highlight(text, X),
            "drivers": self._drivers(X),
            "known_terms": int(X.nnz),
        }

    def predict_many(self, texts: list) -> list:
        cleaned = [clean_text(t)[:MAX_CHARS] for t in texts]
        probs = self.model.predict_proba(self.vectorizer.transform(cleaned))[:, 1]
        return [self._verdict(float(p)) for p in probs]

    def predict_csv(self, raw: bytes, max_rows: int = 500) -> dict:
        """Score an uploaded CSV. Looks for a review/text column; compares to `sentiment` if present."""
        text = raw.decode("utf-8-sig", errors="replace")
        reader = csv.DictReader(io.StringIO(text))
        if not reader.fieldnames:
            raise ValueError("The file looks empty. Add a header row with a 'review' column.")

        columns = {name.strip().lower(): name for name in reader.fieldnames if name}
        review_col = next(
            (columns[c] for c in ("review", "text", "comment", "content", "body") if c in columns),
            reader.fieldnames[0],
        )
        label_col = columns.get("sentiment") or columns.get("label")

        rows = []
        for row in reader:
            value = (row.get(review_col) or "").strip()
            if value:
                rows.append((value, (row.get(label_col) or "").strip().lower() if label_col else ""))
            if len(rows) >= max_rows:
                break
        if not rows:
            raise ValueError(f"No reviews found in column '{review_col}'.")

        verdicts = self.predict_many([r[0] for r in rows])
        positive = sum(v["label"] == "positive" for v in verdicts)

        accuracy = None
        if label_col:
            truth = {"positive": "positive", "1": "positive", "negative": "negative", "0": "negative"}
            pairs = [(truth.get(r[1]), v["label"]) for r, v in zip(rows, verdicts) if truth.get(r[1])]
            if pairs:
                accuracy = round(sum(a == b for a, b in pairs) / len(pairs), 4)

        return {
            "column": review_col,
            "total": len(rows),
            "positive": positive,
            "negative": len(rows) - positive,
            "mean_positive": round(float(np.mean([v["positive"] for v in verdicts])), 4),
            "accuracy": accuracy,
            "truncated": len(rows) >= max_rows,
            "rows": [
                {
                    "text": (text_[:160] + "…") if len(text_) > 160 else text_,
                    "label": v["label"],
                    "summary": v["summary"],
                    "positive": v["positive"],
                    "actual": actual or None,
                }
                for (text_, actual), v in zip(rows[:100], verdicts[:100])
            ],
        }
