import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report

from sklearn.model_selection import train_test_split
import pandas as pd
import re


true = pd.read_csv('True.csv')
fake = pd.read_csv('Fake.csv')

true['label'] = 1
fake['label'] = 0

news = pd.concat([fake,true], axis = 0)

news = news.drop(['title', 'subject', 'date'], axis= 1)
news = news.sample(frac = 1)
news.reset_index(inplace = True)
news.drop(['index'], axis= 1, inplace = True)


def wordopt(text):
  text = text.lower()

  text = re.sub(r'https?://\S+|www\.\S+', '',text)

  text = re.sub(r'<.*?>', '',text)

  text = re.sub(r'[^\w\s]', '', text)

  text = re.sub(r'\d', '', text)

  text = re.sub(r'\n', '', text)

  return text


news['text'] = news['text'].apply(wordopt)

x = news['text']
y = news['label']


x_train,x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3)
vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)


LR = LogisticRegression()
LR.fit(xv_train, y_train)
pred_lr = LR.predict(xv_test)
LR.score(xv_test, y_test)
print(classification_report(y_test, pred_lr))

def output_label(n):
  if n == 0:
    return "It is fake news"
  elif n ==1:
    return "It is real news"
  
def manual_testing(news):
  testing_news = {"text": [news]}
  new_def_test = pd.DataFrame(testing_news)
  new_def_test["text"] = new_def_test['text'].apply(wordopt)
  new_x_test = new_def_test["text"]
  new_xv_test = vectorization.transform(new_x_test)
  pred_lr = LR.predict(new_xv_test)


  return "\n\nLR Prediction: {}".format(
      output_label(pred_lr[0]))




pickle.dump(LR, open('model.pkl', 'wb'))
pickle.dump(vectorization, open('vectorizer.pkl', 'wb'))