import streamlit as st
import pickle
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="IMDB Sentiment Analysis",
    page_icon="🎬",
    layout="centered"
)

# ----------------------------
# Download NLTK Resources
# ----------------------------
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

# ----------------------------
# Load Model & Vectorizer
# ----------------------------
@st.cache_resource
def load_resources():
    with open("tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    with open("logistic_regression_model.pkl", "rb") as f:
        model = pickle.load(f)

    return vectorizer, model

try:
    tfidf_vectorizer, model = load_resources()
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# ----------------------------
# Preprocessing
# ----------------------------
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub(r"\w*\d\w*", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [stemmer.stem(word) for word in tokens]

    return " ".join(tokens)

# ----------------------------
# App UI
# ----------------------------
st.title("🎬 IMDB Movie Review Sentiment Analysis")

st.write("""
Enter a movie review below and the model will predict whether the sentiment is **Positive** or **Negative**.
""")

review = st.text_area(
    "Movie Review",
    height=200,
    placeholder="Type your movie review here..."
)

if st.button("Analyze Sentiment"):

    if not review.strip():
        st.warning("Please enter a movie review.")
    else:

        with st.spinner("Analyzing review..."):

            processed_review = preprocess(review)

            review_vector = tfidf_vectorizer.transform([processed_review])

            prediction = model.predict(review_vector)[0]

            confidence = None
            if hasattr(model, "predict_proba"):
                confidence = model.predict_proba(review_vector)[0]

        st.markdown("---")

        if prediction == 1:
            st.success("😊 **Positive Review**")
        else:
            st.error("😞 **Negative Review**")

        if confidence is not None:
            if prediction == 1:
                st.metric(
                    "Confidence",
                    f"{confidence[1]*100:.2f}%"
                )
            else:
                st.metric(
                    "Confidence",
                    f"{confidence[0]*100:.2f}%"
                )

        with st.expander("Processed Text"):
            st.write(processed_review)

        with st.expander("Original Review"):
            st.write(review)

st.markdown("---")
st.caption(
    "Built with Streamlit • Logistic Regression • TF-IDF Vectorizer"
)
