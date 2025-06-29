import os
import pickle
import string
import pandas as pd
import numpy as np
from rank_bm25 import BM25Okapi
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import time
import shutil

# KONFIGURASI FILE
csv_file_path = "./klikdokter_articles_multi_category.csv"
cache_file = "./bm25_cache_artikel.pkl"

# STOPWORDS
factory = StopWordRemoverFactory()
stop_words = set(factory.get_stop_words())

# PREPROCESSING
def preprocess_text(text):
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

def combine_columns(row):
    return f"{row['title']} {row['content']}"

# BUILD BM25 
def build_bm25_cache():
    # print("Membangun model BM25...")
    data = pd.read_csv(csv_file_path)
    data['combined'] = data.apply(combine_columns, axis=1)
    documents = data['combined'].fillna("").apply(preprocess_text).tolist()

    # Simpan metadata lengkap
    metadata = data[['title', 'content', 'link', 'category']].fillna("").values.tolist()

    tokenized_corpus = [doc.split() for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus)

    # Simpan cache
    with open(cache_file, 'wb') as f:
        pickle.dump((bm25, metadata), f)
    # print("BM25 cache berhasil disimpan.")
    return bm25, metadata

# LOAD BM25
def load_bm25():
    # Hapus cache jika dataset berubah
    if os.path.exists(cache_file):
        csv_mtime = os.path.getmtime(csv_file_path)
        cache_mtime = os.path.getmtime(cache_file)
        if csv_mtime > cache_mtime:
            try:
                os.remove(cache_file)
            except PermissionError:
                # print("File cache sedang digunakan. Menunggu dan mencoba lagi...")
                time.sleep(2)
                try:
                    os.remove(cache_file)
                except Exception as e:
                    # print("Gagal menghapus cache lama:", e)
                    shutil.move(cache_file, cache_file + ".bak")
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            bm25, metadata = pickle.load(f)
        return bm25, metadata
    return build_bm25_cache()

bm25, metadata = load_bm25()

# CARI ARTIKEL
def find_matching_articles(query, top_k=10):
    cleaned = preprocess_text(query)
    tokenized = cleaned.split()
    scores = bm25.get_scores(tokenized)
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        if scores[idx] < 0.1:
            continue
        max_score = max(scores.max(), 1)
        confidence = min(scores[idx] / max_score, 1)
        title, content, link, category = metadata[idx]
        results.append({
            "title": title,
            "content": content,
            "link": link,
            "category": category,
            "score": scores[idx],
            "confidence": confidence
        })
    return results
