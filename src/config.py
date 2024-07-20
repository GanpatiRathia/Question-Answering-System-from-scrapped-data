import os
import nltk
import streamlit as st
from transformers import AutoTokenizer, AutoModel
import google.generativeai as genai
from dotenv import load_dotenv

# Ensure NLTK data is downloaded
nltk_data_path = os.path.join(os.path.expanduser("~"), "nltk_data")
if not os.path.exists(nltk_data_path):
    nltk.download('wordnet')
    nltk.download('punkt')
    nltk.download('stopwords')

# Load environment variables from .env file
load_dotenv()

# Configure Google API key and LLM model
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if GOOGLE_API_KEY is None:
    raise ValueError("GOOGLE_API_KEY environment variable is not set")

genai.configure(api_key=GOOGLE_API_KEY)
llm_model = genai.GenerativeModel(
    model_name='gemini-1.5-flash-latest',
    safety_settings=[
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"}
    ]
)

@st.cache_resource
def load_bert_model():
    """Loads and returns the BERT tokenizer and model."""
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = AutoModel.from_pretrained('bert-base-uncased')
    model.eval()
    return tokenizer, model

tokenizer, model = load_bert_model()