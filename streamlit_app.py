import streamlit as st
import joblib
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from wordcloud import WordCloud

# Set NLTK data directory
nltk.data.path.append(os.path.join(os.getcwd(), 'nltk_data'))

# Download NLTK stopwords and punkt tokenizer data
nltk.download('stopwords', download_dir='nltk_data')
nltk.download('punkt', download_dir='nltk_data')
nltk.download('wordnet', download_dir='nltk_data')

# Load the saved model and vectorizer
model = joblib.load('models/sentiment_model.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')

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

def predict_sentiment(text):
    cleaned_text = clean_text(text)
    without_stopwords = remove_stopwords(cleaned_text)
    lemmatized = lemmatize_text(without_stopwords)
    text_vect = vectorizer.transform([lemmatized])
    prediction = model.predict(text_vect)
    return prediction[0]

# Function to generate word cloud
def generate_wordcloud(text):
    cleaned_text = clean_text(text)
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(cleaned_text)
    st.image(wordcloud.to_array())

# Streamlit app
def main():
    # Customizing Streamlit layout
    st.set_page_config(
        page_title="Sentiment Analysis Predictor",
        page_icon=":sunny:",
        layout="centered",
        initial_sidebar_state="expanded"
    )

    # Custom styles
    st.markdown(
        r"""
        <style>
            html {
                font-family: 'Roboto', sans-serif;
            }
            .reportview-container {
                background: url(r"C:\Users\Admin\Desktop\Projects\Sentiment Analysis of Real-time Flipkart Product Reviews\reviews_badminton\download.png");
                background-size: cover;
                color: #4f8bf9;
            }
            .sidebar .sidebar-content {
                background: #6c757d;
                color: white;
            }
            .stButton>button {
                color: white;
                border-radius: 10px;
                border: 2px solid white;
                background-color: #FF4B4B;
                transition: background-color 0.3s, transform 0.3s;
            }
            .stButton>button:hover {
                background-color: #FFD700;
                transform: scale(1.1);
            }
            .stTextArea>textarea {
                border-radius: 10px;
                border: 2px solid #FFD700;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("Sentiment Analysis Predictor")

    # Input text area for user input
    text = st.text_area("Enter your review here:", "")

    # Predict button
    if st.button("Predict"):
        if text:
            prediction = predict_sentiment(text)
            if prediction == 1:
                st.markdown(f'<h2 style="color:green;">Sentiment: Positive 😊</h2>', unsafe_allow_html=True)
            elif prediction == 0:
                st.markdown(f'<h2 style="color:red;">Sentiment: Negative 😞</h2>', unsafe_allow_html=True)
            else:
                st.info("Sentiment: Neutral or Other")
            # Generate word cloud for the input text
            generate_wordcloud(text)
        else:
            st.warning("Please enter a review.")

if __name__ == "__main__":
    main()
