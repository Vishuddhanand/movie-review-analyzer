"""
app.py  -  Flask web app for the Movie Review Analyzer

Run locally:      python app.py            ->  http://127.0.0.1:5000
Run in production: gunicorn app:app        (see Procfile / Dockerfile)
"""

import os
import random

from flask import Flask, jsonify, render_template, request

from nlp_utils import MAX_CHARS, analyze_text
from sentiment import ModelNotFound, SentimentEngine

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024  # 2 MB upload limit
app.config["JSON_SORT_KEYS"] = False

engine = SentimentEngine()
analyze_text("Warm-up run so the first visitor does not wait for NLTK to load.")


# ------------------------------------------------------------------ pages
@app.get("/")
def index():
    return render_template("index.html", max_chars=MAX_CHARS)


@app.get("/healthz")
def health():
    return jsonify(status="ok")


# -------------------------------------------------------------------- API
@app.post("/api/analyze")
def api_analyze():
    payload = request.get_json(silent=True) or {}
    text = str(payload.get("text", "")).strip()
    if len(text) < 10:
        return jsonify(error="Write at least a sentence (10 characters) to analyze."), 400

    nlp = analyze_text(text)
    sentiment = engine.predict(nlp["text"])
    return jsonify(sentiment=sentiment, nlp=nlp, truncated=len(text) > MAX_CHARS)


@app.post("/api/batch")
def api_batch():
    upload = request.files.get("file")
    if upload is None or not upload.filename:
        return jsonify(error="Choose a CSV file first."), 400
    try:
        return jsonify(engine.predict_csv(upload.read()))
    except ValueError as exc:
        return jsonify(error=str(exc)), 400


@app.get("/api/model")
def api_model():
    return jsonify(engine.metrics)


@app.get("/api/sample")
def api_sample():
    if not engine.samples:
        return jsonify(error="No samples available."), 404
    return jsonify(random.choice(engine.samples))


# ----------------------------------------------------------------- errors
@app.errorhandler(413)
def too_large(_):
    return jsonify(error="That file is larger than 2 MB."), 413


@app.errorhandler(500)
def server_error(_):
    return jsonify(error="Something went wrong while analyzing that text."), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=os.environ.get("FLASK_DEBUG") == "1")
