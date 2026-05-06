# News Predictor
Website: https://mynewspredictor.streamlit.app/
A machine learning web app that detects whether a news article is real or fake using Logistic Regression and TF-IDF vectorization.

## How It Works

1. Text is cleaned (lowercased, URLs/HTML/punctuation/digits removed)
2. A TF-IDF vectorizer converts the text into numerical features
3. A Logistic Regression model predicts real (1) or fake (0)

## Project Structure

```
newspredict/
├── app.py          # Streamlit web app
├── train.py        # Model training script
├── model.pkl       # Trained Logistic Regression model
├── vectorizer.pkl  # Fitted TF-IDF vectorizer
├── True.csv        # Real news dataset
└── Fake.csv        # Fake news dataset
```

## Dataset

`True.csv` and `Fake.csv` were sourced from [GeeksforGeeks](https://www.geeksforgeeks.org/).

## Setup

```bash
pip install streamlit scikit-learn pandas
```

Train the model (only needed once):

```bash
python train.py
```

Run the app:

```bash
streamlit run app.py
```

## Usage

Paste any news article into the text box and click **Analyze**. The app will tell you whether the article is likely real or fake.
