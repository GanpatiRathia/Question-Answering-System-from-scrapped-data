import streamlit as st
import json
import nltk
from nltk.corpus import wordnet, stopwords
from nltk.tokenize import word_tokenize
from pymilvus import connections, Collection, utility
from multiprocessing import Process, Queue
import google.generativeai as genai
from config import tokenizer, model, llm_model  # Ensure correct import
import bm25s
import torch
from dotenv import load_dotenv
import os

# Ensure NLTK data is downloaded
nltk_data_path = os.path.join(os.path.expanduser("~"), "nltk_data")
if not os.path.exists(nltk_data_path):
    nltk.download('wordnet')
    nltk.download('punkt')
    nltk.download('stopwords')

# Load environment variables from .env file
load_dotenv()

def load_scraped_data():
    """Loads previously scraped data from '../data/scraped_data_for_1000_pages.json'."""
    try:
        with open('../data/scraped_data_for_1000_pages.json', 'r', encoding='utf-8') as f:
            scraped_data = json.load(f)
        return scraped_data
    except FileNotFoundError:
        return []

def query_expansion(query):
    """Expands the query using synonyms from WordNet."""
    synonyms = set()
    for syn in wordnet.synsets(query):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return list(synonyms)

def bert_based_retrieval(collection, query, tokenizer, model):
    """Executes BERT-based retrieval on a Milvus collection."""
    inputs = tokenizer(query, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        query_embedding = model(**inputs).last_hidden_state.mean(dim=1).cpu().numpy()
    search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
    results = collection.search([query_embedding[0]], "embedding", param=search_params, limit=100, output_fields=["text"])
    retrieved_texts = [result.entity.get("text") for result in results[0]]
    return retrieved_texts

def bm25_re_rank(retrieved_texts, query):
    """Applies BM25 re-ranking on retrieved texts."""
    stop_words = set(stopwords.words('english'))
    corpus_tokens = [word_tokenize(doc.lower()) for doc in retrieved_texts]
    corpus_tokens = [[word for word in tokens if word.isalnum() and word not in stop_words] for tokens in corpus_tokens]
    
    query_tokens = word_tokenize(query.lower())
    query_tokens = [word for word in query_tokens if word is alnum() and word not in stop_words]
    
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)
    
    results, scores = retriever.retrieve(query_tokens, k=len(retrieved_texts))
    
    ranked_texts = [retrieved_texts[idx] for idx in results[0]]
    return ranked_texts

def main():
    """Main function to run the Streamlit application for Retrieval-Augmented Generation."""    

    st.subheader("Retrieve Documents")
    query = st.text_input("Enter your query:")
    
    if st.button("Retrieve"):
        if query:
            connections.connect("default", host="localhost", port="19530")
            if not utility.has_collection("topic_chunks"):
                st.warning("Collection 'topic_chunks' does not exist")
                return

            collection = Collection("topic_chunks")
            collection.load()

            expanded_query_terms = query_expansion(query)
            expanded_query = " ".join(expanded_query_terms)
            retrieved_texts = bert_based_retrieval(collection, expanded_query, tokenizer, model)
            
            if not retrieved_texts:
                st.warning("No texts retrieved from BERT.")
                return

            ranked_texts = bm25_re_rank(retrieved_texts, query)

            if not ranked_texts:
                st.warning("No texts retrieved after BM25 re-ranking.")
                return

            try:
                documents = [{"text": text} for text in ranked_texts]
                prompt = f"For query: '{query}', refer to the following documents: {documents} answer the question with citation from given reference and detailed information."
                llm_response = llm_model.generate_content(prompt)
                st.subheader("LLM Response:")
                st.write(llm_response.text)
            except Exception as e:
                st.error(f"Error generating response from LLM: {e}")
        else:
            st.warning("Please enter a query.")

if __name__ == "__main__":
    main()