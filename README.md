# ReviewLens: Movie Review Sentiment Analysis

A complete NLP web app built from the `movie_nlp.ipynb` lab notebook. Paste a movie review and get:

- **Verdict**: positive, negative or mixed, with probabilities
- **Word-by-word explanation**: every word shaded by how much it pushed the prediction
- **Language breakdown**: language detection, word and sentence counts, tokens, stemming vs lemmatization, parts of speech, named entities, common n-grams
- **CSV batch mode**: score up to 500 reviews at once
- **Model report**: accuracy, confusion matrix, model comparison, strongest words

**Model:** TF-IDF (1 and 2-word phrases, 150,000 features) + Logistic Regression, trained on the IMDB 50K dataset.
**Accuracy: 91.4%** on 9,917 held-out reviews, versus 85.8% for the LSA approach in the notebook.

---

## 1. Run it locally

You need Python 3.11 or newer.

```bash
cd movie-review-analyzer

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
python app.py
```

Open **http://127.0.0.1:5000**

The trained model files (`models/*.pkl`) and NLTK data (`nltk_data/`) are already in the folder, so nothing else needs downloading.

## 2. Folder structure

```
movie-review-analyzer/
├── app.py                  Flask app: page + JSON API
├── sentiment.py            loads the .pkl files, predicts, explains predictions
├── nlp_utils.py            tokenizing, stemming, POS, NER, n-grams, language detection
├── train_model.py          trains the model and writes models/*
├── models/
│   ├── vectorizer.pkl      TF-IDF vectorizer
│   ├── sentiment_model.pkl Logistic Regression classifier
│   ├── metrics.json        accuracy, confusion matrix, comparison, top words
│   └── samples.json        reviews for the "Try a sample review" button
├── templates/index.html    the page
├── static/                 style.css, app.js, example_reviews.csv
├── nltk_data/              bundled NLTK resources (works offline)
├── data/                   put "IMDB Dataset.csv" here only if you retrain
├── requirements.txt        runtime dependencies (pinned)
├── requirements-train.txt  adds pandas for retraining
├── Dockerfile  Procfile  render.yaml  .python-version
```

## 3. Retrain the model (optional)

The `.pkl` files are already trained. To rebuild them, for example from your own copy of the dataset:

```bash
pip install -r requirements-train.txt
# put "IMDB Dataset.csv" in the data/ folder (same file the notebook uses)
python train_model.py
```

It takes about 2 minutes and rewrites everything in `models/`. Use `--sample 5000` for a quick test run, or `--data "path/to/file.csv"` to read the CSV from elsewhere.

> **If you see a scikit-learn version warning or a pickle error** after changing package versions, just run `python train_model.py` again. Pickle files only load reliably on the same scikit-learn version that created them, which is why `requirements.txt` pins it.

## 4. Deploy

### Option A: Render (free, easiest)
1. Push this folder to a GitHub repository (the `.pkl` files are small, so commit them).
2. On [render.com](https://render.com): **New + > Blueprint** and choose the repo. `render.yaml` configures everything.
   Or **New + > Web Service** with build command `pip install -r requirements.txt` and start command `gunicorn app:app --workers 1 --threads 4 --timeout 120 --bind 0.0.0.0:$PORT`.
3. Wait for the build. The free tier sleeps when idle, so the first visit after a pause takes about 30 seconds.

### Option B: Railway / Heroku
Both read the `Procfile`. Connect the GitHub repo and deploy.

### Option C: Docker (any cloud, VPS, or locally)
```bash
docker build -t reviewlens .
docker run -p 8000:8000 reviewlens
```
Open http://localhost:8000. This image works on Google Cloud Run, AWS App Runner, Azure Container Apps and Fly.io.

### Option D: Hugging Face Spaces (Docker)
Create a Space with the **Docker** SDK, upload this folder, and set `PORT=7860` in the Space variables. The Dockerfile already reads `PORT`.

The app uses roughly 450-500 MB of RAM once warmed up (measured: 464 MB), because scikit-learn, NLTK and the 150,000-term vocabulary all live in memory. It runs one gunicorn worker with 4 threads. A 512 MB free tier is tight; if the host kills it for memory, choose a 1 GB plan.

## 5. API (if you want to call it from another app)

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/api/analyze` | body `{"text": "..."}`, returns sentiment plus the full NLP breakdown |
| POST | `/api/batch` | multipart upload, field `file` (CSV), scores up to 500 rows |
| GET | `/api/model` | training metrics |
| GET | `/api/sample` | a random sample review |
| GET | `/healthz` | health check |

```bash
curl -X POST http://127.0.0.1:5000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "A brilliant, moving film with superb acting."}'
```

## 6. How the notebook maps to the app

| Notebook experiment | Where it lives in the app |
|---|---|
| Tokenization, word and sentence counts, language detection | Language breakdown > Overview, Tokens and roots |
| Stemming vs lemmatization | Language breakdown > Tokens and roots |
| POS tagging | Language breakdown > Parts of speech |
| Named entity recognition | Language breakdown > Names |
| TF-IDF | The model's features, and "Words that signal positive/negative" |
| N-grams | Language breakdown > Common phrases |
| LSA + Logistic Regression sentiment | Benchmarked in "Models compared"; scatter plot in "About the model" |
| Word2Vec | Not used. On the full dataset TF-IDF beats averaged word vectors, and it keeps the app small |
| Wikipedia scraping | Not included. It is a separate exercise from review analysis |

## 7. Honest limitations

- The model reads word patterns, not meaning. **Sarcasm and tricky negation can fool it** ("not bad at all" may score negative).
- Trained on IMDB reviews written in English, so it works best on that kind of text.
- The "Try a sample review" button picks from reviews the model gets right. The example CSV is random and unfiltered, so its accuracy is realistic.
- Reviews longer than 5,000 characters are truncated.
