import nltk
from flask import Flask, render_template, request
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the dataset
df = pd.read_csv('data.csv')

# Drop rows with missing values in 'Place of Review' column
df.dropna(subset=['Place of Review'], inplace=True)

df['Review text'].fillna('', inplace=True)
df['Review Title'].fillna('', inplace=True)
df['Reviewer Name'].fillna('', inplace=True)
df['Month'].fillna('Unknown', inplace=True)

# Define thresholds for positive and negative sentiment
positive_threshold = 3.5
negative_threshold = 2.5

# Infer sentiment based on rating
def infer_sentiment(rating):
    if rating >= positive_threshold:
        return 1  # Positive sentiment
    elif rating <= negative_threshold:
        return 0  # Negative sentiment
    else:
        return -1  # Neutral sentiment or other

# Apply sentiment inference to the dataset
df['Sentiment'] = df['Ratings'].apply(infer_sentiment)

# Preprocessing functions
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters and punctuation
    text = text.lower()  # Convert text to lowercase
    return text

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    filtered_text = [word for word in tokens if word not in stop_words]
    return " ".join(filtered_text)

def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    lemmatized_text = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(lemmatized_text)

# Apply preprocessing to the dataset
df['Review text'] = df['Review text'].apply(clean_text)
df['Review text'] = df['Review text'].apply(remove_stopwords)
df['Review text'] = df['Review text'].apply(lemmatize_text)

# Train the model using the entire dataset
vectorizer = TfidfVectorizer(max_features=1000)
X_vect = vectorizer.fit_transform(df['Review text'])
y = df['Sentiment']

# Updated: Use Random Forest Classifier
model = RandomForestClassifier()
model.fit(X_vect, y)

# Predict on the entire dataset
y_pred = model.predict(X_vect)

# Compute F1 score
f1 = f1_score(y, y_pred, average='weighted')
print("F1 Score:", f1)

# Function to predict sentiment
def predict_sentiment(text):
    text = clean_text(text)
    text = remove_stopwords(text)
    text = lemmatize_text(text)
    text_vect = vectorizer.transform([text])
    prediction = model.predict(text_vect)
    return prediction[0]

# Flask routes go here, if needed

