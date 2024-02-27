import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from tiktoken import get_encoding

def top_related_tfidf(key, refs, k=5, indices=False):  # TF-IDF
    vectorizer = TfidfVectorizer()
    docs = [doc for title, url, doc in refs]
    tfidf_matrix = vectorizer.fit_transform(docs)
    key_vector = vectorizer.transform([key])
    similarities = cosine_similarity(key_vector, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[-k:][::-1]
    if indices: return top_indices
    top_related_refs = [refs[i] for i in top_indices]
    return top_related_refs

def top_related_bm25(key, refs, k=5, indices=False):  # BM25
    docs = [nltk.word_tokenize(doc) for title, url, doc in refs]
    bm25 = BM25Okapi(docs)
    doc_scores = bm25.get_scores(nltk.word_tokenize(key))
    top_indices = doc_scores.argsort()[-k:][::-1]
    if indices: return top_indices
    top_related_refs = [refs[i] for i in top_indices]
    return top_related_refs

def get_top_related_docs(key, refs, k=5, method="tfidf", indices=False):
    func = {"tfidf": top_related_tfidf, "bm25": top_related_bm25}
    return func[method.lower()](key, refs, k, indices)

max_tokens = {
    "gpt-3.5-turbo": 4096,
    "gpt-3.5-turbo-0613": 4096,
    "gpt-3.5-turbo-16k": 16385,
    "gpt-3.5-turbo-16k-0613": 16385,
    "gpt-4": 8192,
    "gpt-4-0613": 8192,
    "gpt-4-32k": 32768,
    "gpt-4-32k-0613": 32768,
}

def count_tokens(messages):
    encoding = get_encoding("cl100k_base")
    tokens_per_message, tokens_per_name = 3, 1
    num_tokens = 3
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name": num_tokens += tokens_per_name
    return num_tokens