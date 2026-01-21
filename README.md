# Text_Mining_with_Topic_and_Sentiment_Analysis

Topic and sentiment analysis of tweets with #donaldtrump during the 2020 US election period.

## Dataset

Dataset (download and skip API and language detection parts if you use this file directly):
https://drive.google.com/file/d/1nbB92NIDPdGZq4mQjLF1TDW-vCE9g9Uj/view?usp=sharing

The raw dataset is multilingual. We keep only English tweets for topic modeling and sentiment analysis.

## Setup (uv)

1. Install uv (see https://docs.astral.sh/uv/).
2. Create the environment and install dependencies:
   - `uv sync`
3. Start Jupyter from the uv environment:
   - `uv run jupyter lab`

## Notebooks

- `API.ipynb`: Download the Kaggle dataset via the Kaggle API.
- `LANGUAGE_DETECTION.ipynb`: Detect language with the lid.176 model and filter to English tweets.
- `full_project.ipynb`: End-to-end pipeline (data inspection, preprocessing, topic modeling, sentiment analysis, and joint topic-sentiment analysis).

## Workflow (full_project.ipynb)

1. Data overview and quality checks (size, columns, missing text, duplicates/retweets).
2. Preprocessing:
   - Basic cleaning (URLs, mentions, hashtags formatting, lowercasing, whitespace).
   - Remove very short tweets.
   - Stopword handling (generic and task-specific).
   - Two text versions:
     - `text_for_topics` for stronger cleaning.
     - `text_for_sentiment` for lighter cleaning.
3. Quick EDA: common unigrams/bigrams and a word cloud.
4. Topic modeling:
   - TF-IDF vectorization.
   - NMF topic model with interpretable topics.
   - Assign topics to tweets and inspect example tweets.
5. Sentiment analysis (VADER):
   - Compound score and 3-class labels.
   - Overall sentiment distribution and sanity checks.
6. Supervised sentiment:
   - Weak labels from high-confidence VADER scores.
   - Models: TF-IDF + Logistic Regression, LinearSVC, ComplementNB.
   - Apply best model (LinearSVC) to full dataset.
7. Joint analysis: sentiment distribution within topics and topic-level sentiment comparison.

## Research questions

1. What are the dominant themes in Trump-related tweets?
2. What is the overall balance of positive, neutral, and negative sentiment?
3. Which themes are most positive or most negative?

## Notes and limitations

- Supervised sentiment is weakly supervised (no human ground truth).
- Political tweets contain sarcasm and noisy signals.
- Topic model evaluation is interpretability-based.
