
import streamlit as st
import pickle
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Ensure NLTK data is available (for local execution)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# --- Load the trained model and TF-IDF vectorizer ---
@st.cache_resource
def load_resources():
    # Load the TF-IDF Vectorizer
    with open('tfidf_vectorizer.pkl', 'rb') as file:
        loaded_tfidf_vectorizer = pickle.load(file)

    # Load the best performing model (Logistic Regression)
    with open('logistic_regression_model.pkl', 'rb') as file:
        loaded_model = pickle.load(file)
    
    return loaded_tfidf_vectorizer, loaded_model

tfidf_vectorizer, model = load_resources()

# --- Text Preprocessing Functions (copied from notebook) ---
def clean_text(text):
    text = text.lower() # Lowercasing
    text = re.sub(r'<.*?>', '', text) # Remove HTML tags
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text) # Remove punctuation
    text = re.sub(r'\w*\d\w*', '', text) # Remove words containing numbers
    text = re.sub(r'\s+', ' ', text).strip() # Remove extra spaces
    return text

def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    return [word for word in tokens if word not in stop_words]

def apply_stemming(tokens):
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in tokens]

# --- Streamlit Application ---
st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review below to predict its sentiment (positive or negative).')

user_input = st.text_area('Enter your review here:', height=150)

if st.button('Analyze Sentiment'):
    if user_input:
        # Preprocess the input text
        cleaned_text = clean_text(user_input)
        tokenized_text = word_tokenize(cleaned_text)
        filtered_text = remove_stopwords(tokenized_text)
        stemmed_text = apply_stemming(filtered_text)
        processed_review = ' '.join(stemmed_text)

        # Transform the processed text using the loaded TF-IDF vectorizer
        # Ensure the input to transform is a list-like object
        text_tfidf = tfidf_vectorizer.transform([processed_review])

        # Make prediction
        prediction = model.predict(text_tfidf)

        # Decode the prediction
        sentiment_label = 'Positive' if prediction[0] == 1 else 'Negative'

        st.subheader('Prediction:')
        if sentiment_label == 'Positive':
            st.success(f'The review is: **{sentiment_label}** 😃')
        else:
            st.error(f'The review is: **{sentiment_label}** 😞')
    else:
        st.warning('Please enter some text to analyze.')
