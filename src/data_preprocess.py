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
    logging.info("Loading scraped data from %s", file_path)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            scraped_data = json.load(f)
        return scraped_data
    except FileNotFoundError:
        logging.error("File not found: %s", file_path)
        return []

def preprocess_text(text):
    text = re.sub(r'\S*\d\S*', '', text).strip()
    text = re.sub(r'\s+', ' ', text).strip()
    return [word for word in simple_preprocess(text)]

def chunk_by_similarity(sentences, model, tokenizer, n_clusters=10):
    def get_embeddings(sentences, model, tokenizer, batch_size=16):
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
    
    return clusters, embeddings

def save_embeddings(file_path, embeddings, urls, texts):
    logging.info("Saving embeddings to %s", file_path)
    data = [{"url": url, "embedding": embedding.tolist(), "text": text[:65535]} for url, embedding, text in zip(urls, embeddings, texts)]
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def load_embeddings(file_path):
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
    logging.info("Connecting to Milvus")

    connections.connect("default", host="localhost", port="19530")

    if utility.has_collection("topic_chunks"):
        utility.drop_collection("topic_chunks")

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
        FieldSchema(name="url", dtype=DataType.VARCHAR, max_length=2048),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535)
    ]

    schema = CollectionSchema(fields, "Topic chunks collection")
    collection = Collection("topic_chunks", schema)
    index_params = {"index_type": "IVF_FLAT", "params": {"nlist": 128}, "metric_type": "L2"}
    collection.create_index("embedding", index_params)

    return collection

def insert_into_milvus(collection, embeddings, urls, texts, batch_size=100):
    logging.info("Inserting data into Milvus")
    try:
        for i in range(0, len(embeddings), batch_size):
            batch_embeddings = embeddings[i:i + batch_size]
            batch_urls = urls[i:i + batch_size]
            batch_texts = [text[:65535] for text in texts[i:i + batch_size]]
            entities = [
                {"name": "embedding", "type": DataType.FLOAT_VECTOR, "values": batch_embeddings},
                {"name": "url", "type": DataType.VARCHAR, "values": batch_urls},
                {"name": "text", "type": DataType.VARCHAR, "values": batch_texts}
            ]
            collection.insert(entities)
            collection.flush()
        logging.info("Data inserted into Milvus successfully.")
    except Exception as e:
        logging.error("Error inserting data into Milvus: %s", e)

def main(model, tokenizer):
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