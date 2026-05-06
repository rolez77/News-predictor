import streamlit as st
import pickle
import re

def wordopt(text):
  text = text.lower()

  text = re.sub(r'https?://\S+|www\.\S+', '',text)

  text = re.sub(r'<.*?>', '',text)

  text = re.sub(r'[^\w\s]', '', text)

  text = re.sub(r'\d', '', text)

  text = re.sub(r'\n', '', text)

  return text
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

st.title("Fake News Detection")

article = st.text_area("Enter the news article here:", height=300 )


if st.button("Analyze"):

  cleaned = wordopt(article)
  vectorized = vectorizer.transform([cleaned])
  prediction = model.predict(vectorized)[0]

  if prediction == 1:
    st.success("The news is likely REAL.")
  else:
    st.error("The news is likely FAKE.")