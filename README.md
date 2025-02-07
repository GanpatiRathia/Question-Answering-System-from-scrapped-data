# Question-Answering System from Scrapped Data

This project involves creating a question-answering system using scraped data. The system performs data chunking, vector database creation, retrieval, re-ranking, and uses advanced techniques for generating answers.

## Prerequisites

- Python 3.7 or higher
- pip (Python package installer)
- NVIDIA GPU (optional for faster processing)

## Setup Instructions

### 1. Clone the Repository

```sh
git clone <repository_url>
cd Question-Answering-System-from-scrapped-data
```

### 2. Create and Activate a Virtual Environment
```ssh
python -m venv venv
venv\Scripts\activate
```

### 3. Install Required Packages
```ssh
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Create a .env file in the root directory of the project and add your Google API key:

```ssh
GOOGLE_API_KEY=your_google_api_key_here
```

### 5. Download NLTK Data

Ensure that NLTK data is downloaded for tokenization and stopwords:

```ssh
import nltk

nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')
```

### 6. Run for webscrapping 
```ssh
python webscrap.py  
``` 

Output :
![Diagram](imgs/output/webscrap.png)

### 7. Do setup milvus using docker and use yml config

### 8. Run this after webscrapping to Data Chunking and Vector Database Creation,
```ssh
python data_preprocess.py 
```
Output :
![Diagram](imgs/output/preprocess.png)

### 9. Run this after runnning all previous scripts for Retrieval and Re-ranking and UI 
```ssh
streamlit run question-answer-app.py
```
Script Output :
![Diagram](imgs/output/start-app.png)

UI Output :
![Diagram](imgs/output/app.png)