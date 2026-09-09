import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

try:
    nltk.download('punkt_tab')
except:
    pass

data = {
    "Text": [
        "I love this phone",
        "This mobile is excellent",
        "Amazing camera quality",
        "Very good battery",
        "I like this product",
        "I hate this phone",
        "Worst mobile ever",
        "Very bad battery",
        "Poor camera quality",
        "I dislike this product"
    ],
    "Sentiment": [
        "Positive", "Positive", "Positive", "Positive", "Positive",
        "Negative", "Negative", "Negative", "Negative", "Negative"
    ]
}

df = pd.DataFrame(data)

print("Original Dataset")
print(df)

df["Text"] = df["Text"].str.lower()
df["Text"] = df["Text"].str.replace(
    '[{}]'.format(string.punctuation), '', regex=True
)
df["Text"] = df["Text"].str.replace(r'\d+', '', regex=True)
df["Text"] = df["Text"].apply(word_tokenize)
stop_words = set(stopwords.words('english'))
df["Text"] = df["Text"].apply(
    lambda words: [word for word in words if word not in stop_words]
)
stemmer = PorterStemmer()
df["Text"] = df["Text"].apply(
    lambda words: [stemmer.stem(word) for word in words]
)
lemmatizer = WordNetLemmatizer()
df["Text"] = df["Text"].apply(
    lambda words: [lemmatizer.lemmatize(word) for word in words]
)
df["Text"] = df["Text"].apply(lambda words: " ".join(words))

print("\nProcessed Dataset")
print(df)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["Text"])

y = df["Sentiment"]

model = MultinomialNB()
model.fit(X, y)

print("\nModel Trained Successfully")

sentence = "This phone has good camera"

sentence = sentence.lower()
sentence = re.sub(r'[^\w\s]', '', sentence)
sentence = re.sub(r'\d+', '', sentence)

words = word_tokenize(sentence)

words = [word for word in words if word not in stop_words]

words = [stemmer.stem(word) for word in words]

words = [lemmatizer.lemmatize(word) for word in words]

sentence = " ".join(words)

test = vectorizer.transform([sentence])

prediction = model.predict(test)

print("\nTest Sentence:", sentence)
print("Predicted Sentiment:", prediction[0])