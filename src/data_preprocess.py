import json
import re
import logging
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
import torch
from gensim.utils import simple_preprocess
import numpy as np
from sklearn.cluster import KMeans

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_scraped_data(file_path):
    """Loads scraped data from a JSON file.

    Args:
        file_path (str): Path to the JSON file containing scraped data.

    Returns:
        list: A list of scraped data.
    """
    logging.info("Loading scraped data from %s", file_path)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            scraped_data = json.load(f)
        return scraped_data
    except FileNotFoundError:
        logging.error("File not found: %s", file_path)
        return []

def preprocess_text(text):
    """Cleans and preprocesses the input text.

    Args:
        text (str): Raw text to preprocess.

    Returns:
        list: List of preprocessed tokens.
    """
    text = re.sub(r'\S*\d\S*', '', text).strip()
    text = re.sub(r'\s+', ' ', text).strip()
    return [word for word in simple_preprocess(text)]

def chunk_by_similarity(sentences, model, tokenizer, n_clusters=10):
    """Chunks text into clusters based on semantic similarity using a BERT model.

    Args:
        sentences (list): List of sentences to cluster.
        model (transformers.PreTrainedModel): Pre-trained BERT model for embeddings.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer for the BERT model.
        n_clusters (int, optional): Number of clusters. Defaults to 10.

    Returns:
        tuple: Dictionary of clusters and corresponding embeddings.
    """
    
    def get_embeddings(sentences, model, tokenizer, batch_size=16):
        """Generates embeddings for the sentences with batch processing.

        Args:
            sentences (list): List of sentences to generate embeddings for.
            model (transformers.PreTrainedModel): Pre-trained BERT model.
            tokenizer (transformers.PreTrainedTokenizer): Tokenizer for the BERT model.
            batch_size (int, optional): Batch size for embedding generation. Defaults to 16.

        Returns:
            np.ndarray: Array of sentence embeddings.
        """
        all_embeddings = []
        for i in range(0, len(sentences), batch_size):
            batch_sentences = sentences[i:i + batch_size]
            inputs = tokenizer(batch_sentences, return_tensors='pt', padding=True, truncation=True)
            with torch.no_grad():
                outputs = model(**inputs)
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(batch_embeddings)
        return np.vstack(all_embeddings)

    embeddings = get_embeddings(sentences, model, tokenizer)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(embeddings)
    clusters = {i: [] for i in range(n_clusters)}
    for idx, label in enumerate(kmeans.labels_):
        clusters[label].append(sentences[idx])
    
    return clusters, embeddings  # Return embeddings along with clusters

def save_embeddings(file_path, embeddings, urls, texts):
    """Saves embeddings, URLs, and texts to a JSON file.

    Args:
        file_path (str): Path to the JSON file to save embeddings.
        embeddings (list): List of embeddings.
        urls (list): List of URLs.
        texts (list): List of texts.
    """
    logging.info("Saving embeddings to %s", file_path)
    data = [{"url": url, "embedding": embedding.tolist(), "text": text} for url, embedding, text in zip(urls, embeddings, texts)]
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def load_embeddings(file_path):
    """Loads embeddings, URLs, and texts from a JSON file.

    Args:
        file_path (str): Path to the JSON file containing embeddings.

    Returns:
        tuple: Lists of embeddings, URLs, and texts.
    """
    logging.info("Loading embeddings from %s", file_path)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        embeddings = [item['embedding'] for item in data]
        urls = [item['url'] for item in data]
        texts = [item['text'] for item in data]
        return embeddings, urls, texts
    except FileNotFoundError:
        logging.error("File not found: %s", file_path)
        return [], [], []

def connect_milvus():
    """Connects to the Milvus database and creates a collection if it does not exist.

    Returns:
        Collection: The Milvus collection object.
    """
    logging.info("Connecting to Milvus")
    from pymilvus import connections

    # Connect to Milvus server
    connections.connect("default", host="localhost", port="19530")

    # Check if the connection is successful
    if connections.has_connection("default"):
        print("Connected to Milvus server.")
    else:
        print("Failed to connect to Milvus server.")
    if utility.has_collection("topic_chunks"):
        utility.drop_collection("topic_chunks")

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
        FieldSchema(name="url", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535)
    ]

    schema = CollectionSchema(fields, "Topic chunks collection")
    collection = Collection("topic_chunks", schema)
    index_params = {"index_type": "IVF_FLAT", "params": {"nlist": 128}, "metric_type": "L2"}
    collection.create_index("embedding", index_params)

    return collection

def insert_into_milvus(collection, embeddings, urls, texts, batch_size=100):
    """Inserts embeddings, URLs, and texts into the specified Milvus collection.

    Args:
        collection (Collection): The Milvus collection object.
        embeddings (list): List of embeddings.
        urls (list): List of URLs.
        texts (list): List of texts.
        batch_size (int, optional): Batch size for Milvus insertion. Defaults to 100.
    """
    logging.info("Inserting data into Milvus")
    entities = {"embedding": embeddings, "url": urls, "text": [text[:65530] for text in texts]}
    try:
        for i in range(0, len(embeddings), batch_size):
            batch_embeddings = embeddings[i:i + batch_size]
            batch_urls = urls[i:i + batch_size]
            batch_texts = [text[:65530] for text in texts[i:i + batch_size]]
            collection.insert([batch_embeddings, batch_urls, batch_texts])
            collection.flush()
        logging.info("Data inserted into Milvus successfully.")
    except Exception as e:
        logging.error("Error inserting data into Milvus: %s", e)

def main(model, tokenizer):
    """Main function to process scraped data, create embeddings, and insert them into Milvus.

    Args:
        model (transformers.PreTrainedModel): Pre-trained BERT model for embeddings.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer for the BERT model.
    """
    scraped_data = load_scraped_data('../data/scraped_data_for_100_pages.json')
    if scraped_data:
        texts = [data['text'] for data in scraped_data]
        urls = [data['url'] for data in scraped_data]
        
        clusters, embeddings = chunk_by_similarity(texts, model, tokenizer, n_clusters=10)
        
        chunked_texts = []
        for cluster in clusters.values():
            chunked_texts.extend(cluster)
        
        save_embeddings('../data/embeddings.json', embeddings, urls, chunked_texts)
        
        logging.info("Embeddings saved to embeddings.json successfully.")
        embeddings, urls, texts = load_embeddings('../data/embeddings.json')
        if embeddings and urls:
            collection = connect_milvus()
            logging.info("Connection to Milvus successful")
            insert_into_milvus(collection, embeddings, urls, texts)
        else:
            logging.warning("No embeddings available to insert.")
    else:
        logging.warning("No scraped data available.")

if __name__ == "__main__":
    from transformers import BertTokenizer, BertModel

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    main(model, tokenizer)