import re
import os
import math
import json
import html
import numpy as np_cpu
import cupy as np
from html.parser import HTMLParser
from collections import defaultdict

# ─── REUSED FROM BAYES ───────────────────────────────────────────────────────

STOPWORDS = {
    "a","an","the","and","or","but","if","then","else","of","in","on","at","by",
    "for","from","with","to","into","onto","upon","is","are","was","were","be",
    "been","being","do","does","did","have","has","had","this","that","these",
    "those","it","its","as","than","so","such","because","while","although",
    "about","against","between","during","before","after","above","below",
    "again","further","once",
    "you","your","we","our","i","my","me","us","will","can","no","not","all",
    "one","out","just","may","here","more","any","get","now","new","only","please",
    "enron","ect","hou","vince","kaminski","daren","hpl","corp",
    "escapenumber","escapelong","escapenumbermg",
    "td","tr","font","bgcolor","nbsp","href","img","width","height",
    "style","pt","cc","subject","per","nextpart","mime","multipart","content","charset",
    "transfer","encoding","quoted","printable","boundary",
    "listmaster","mailman","listinfo","unsubscribe","linux","ilug","irish","users","group",
    "multipart","format","plain","windows","type","text",
    "base","tbit","tab","decoration","none","multi","part","legal","notice","iso","instead",
    "go","format","type","plain","windows","text",
    "com","org","net","edu","gov","de","uk","ca","au",
    "inc","re","fw","fwd","pm","am","http","www",
    "align","center","index","border","valign","htmlimg","size","u.s"
}

class HTMLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.text_parts = []
    def handle_data(self, data):
        self.text_parts.append(data)
    def get_text(self):
        return " ".join(self.text_parts)

def strip_html(text):
    stripper = HTMLStripper()
    try:
        stripper.feed(html.unescape(text))
        return stripper.get_text()
    except:
        return text

def is_garbage_token(token):
    vowels = set("aeiou")
    alpha_chars = [c for c in token if c.isalpha()]
    digit_chars = [c for c in token if c.isdigit()]
    if len(token) > 1 and len(digit_chars) / len(token) > 0.3:
        return True
    if len(token) <= 2 and any(c.isdigit() for c in token):
        return True
    if len(alpha_chars) >= 3 and not any(c in vowels for c in alpha_chars):
        return True
    return False

def normalize_word(word):
    substitutions = {"0":"o","1":"i","3":"e","4":"a","$":"s","@":"a","5":"s","7":"t"}
    for k, v in substitutions.items():
        word = word.replace(k, v)
    return word

def merge_letter_sequences(tokens):
    merged = []
    buffer = []
    for t in tokens:
        if len(t) == 1 and t.isalpha():
            buffer.append(t)
        else:
            if len(buffer) > 1:
                merged.append("".join(buffer))
            elif buffer:
                merged.extend(buffer)
            buffer = []
            merged.append(t)
    if len(buffer) > 1:
        merged.append("".join(buffer))
    else:
        merged.extend(buffer)
    return merged

def add_ngrams(tokens, n=4):
    ngrams = []
    for k in range(2, n + 1):
        for i in range(len(tokens) - k + 1):
            ngram = "_".join(tokens[i:i + k])
            ngrams.append(ngram)
    return tokens + ngrams

def tokenize_email(email):
    url_pattern = r"(http\S+|www\.\S+)"
    token_pattern = r"[a-zA-Z0-9$@\.]+"
    email = email.lower()
    email = strip_html(email)
    email = re.sub(url_pattern, "<URL>", email)
    email = re.sub(r"(.)\1{2,}", r"\1\1", email)
    tokens = re.findall(token_pattern, email)
    tokens = [t.strip("._-") for t in tokens]
    tokens = [t for t in tokens if len(t) > 1]
    tokens = [t for t in tokens if not ("." in t and not t.replace(".", "").isalpha())]
    tokens = [t for t in tokens if not is_garbage_token(t)]
    tokens = [normalize_word(t) for t in tokens]
    tokens = [t for t in tokens if len(t) > 1]
    tokens = [t for t in tokens if t not in STOPWORDS]
    tokens = [t for t in tokens if not (len(t) <= 3 and t.isalpha())]
    tokens = merge_letter_sequences(tokens)
    tokens = add_ngrams(tokens, n=4)
    tokens = [t for t in tokens if not any(is_garbage_token(part) for part in t.split("_"))]
    return tokens

# ─── FEATURE SELECTION ────────────────────────────────────────────────────────

def build_vocabulary(ham_tokens, spam_tokens, top_k=20000, min_ratio=0.3):
    spam_docs = len(spam_tokens)
    ham_docs = len(ham_tokens)

    spam_df = defaultdict(int)
    ham_df = defaultdict(int)

    for email in spam_tokens:
        for word in set(email):
            spam_df[word] += 1

    for email in ham_tokens:
        for word in set(email):
            ham_df[word] += 1

    vocabulary = set(spam_df) | set(ham_df)
    scores = {}

    for word in vocabulary:
        A = spam_df.get(word, 0)
        C = ham_df.get(word, 0)
        B = spam_docs - A
        D = ham_docs - C
        N = A + B + C + D
        denom = (A + C) * (B + D) * (A + B) * (C + D)
        if denom == 0:
            continue
        chi2 = (N * (A * D - B * C) ** 2) / denom
        scores[word] = chi2

    filtered = {}
    for word, chi2 in scores.items():
        spam_rate = spam_df.get(word, 0) / spam_docs
        ham_rate = ham_df.get(word, 0) / ham_docs
        if ham_rate == 0 or spam_rate == 0:
            filtered[word] = chi2
            continue
        ratio = abs(math.log(spam_rate / ham_rate))
        if ratio >= min_ratio:
            filtered[word] = chi2

    top_features = sorted(filtered, key=filtered.get, reverse=True)[:top_k]
    return top_features

# ─── FEATURE MATRIX (CPU numpy — CuPy can't index with strings) ──────────────

def build_feature_matrix(all_tokens, word_to_idx):
    N = len(all_tokens)
    V = len(word_to_idx)
    X = np_cpu.zeros((N, V), dtype=np_cpu.float32)
    for i, tokens in enumerate(all_tokens):
        for token in tokens:
            if token in word_to_idx:
                X[i, word_to_idx[token]] += 1
    return X

# ─── SIGMOID ──────────────────────────────────────────────────────────────────

def sigmoid(z):
    z = np.clip(z, -30, 30)
    return 1 / (1 + np.exp(-z))

# ─── LOSS ─────────────────────────────────────────────────────────────────────

def compute_loss(y_hat, y):
    y_hat = np.clip(y_hat, 1e-7, 1 - 1e-7)
    return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

# ─── TRAINING ─────────────────────────────────────────────────────────────────

def train(X, y, learning_rate=0.1, iterations=5000, log_every=500):
    N, V = X.shape

    w = np.zeros(V, dtype=np.float32)
    b = np.float32(0.0)

    for i in range(iterations):
        z = X @ w + b
        y_hat = sigmoid(z)

        loss = compute_loss(y_hat, y)

        error = y_hat - y
        dw = (X.T @ error) / N
        db = np.mean(error)

        w -= learning_rate * dw
        b -= learning_rate * db

        if i % log_every == 0 or i == iterations - 1:
            print(f"Iteration {i:5d} | Loss: {float(loss):.4f}")

    return w, b

# ─── LOAD DATA ────────────────────────────────────────────────────────────────

ham_folder = "logisticdatasets/hams-data"
spam_folder = "logisticdatasets/spams-data"

ham_files = [os.path.join(ham_folder, f) for f in os.listdir(ham_folder) if f.endswith(".txt")]
spam_files = [os.path.join(spam_folder, f) for f in os.listdir(spam_folder) if f.endswith(".txt")]

print("Loading and tokenizing emails...")

ham_emails = []
for file_path in ham_files:
    with open(file_path, "r", encoding="latin-1") as f:
        for line in f:
            line = line.strip()
            if line:
                ham_emails.append(line)

spam_emails = []
for file_path in spam_files:
    with open(file_path, "r", encoding="latin-1") as f:
        for line in f:
            line = line.strip()
            if line:
                spam_emails.append(line)

print(f"Ham emails:  {len(ham_emails)}")
print(f"Spam emails: {len(spam_emails)}")

print("Tokenizing...")
ham_tokens = [tokenize_email(e) for e in ham_emails]
spam_tokens = [tokenize_email(e) for e in spam_emails]

# ─── BUILD VOCABULARY ─────────────────────────────────────────────────────────

print("Building vocabulary with chi-square feature selection...")
vocabulary = build_vocabulary(ham_tokens, spam_tokens, top_k=20000)
word_to_idx = {word: idx for idx, word in enumerate(vocabulary)}
print(f"Vocabulary size: {len(vocabulary)}")

# ─── BUILD FEATURE MATRIX ─────────────────────────────────────────────────────

print("Building feature matrix on CPU...")
all_tokens = ham_tokens + spam_tokens
y_cpu = np_cpu.array([0] * len(ham_tokens) + [1] * len(spam_tokens), dtype=np_cpu.float32)
X_cpu = build_feature_matrix(all_tokens, word_to_idx)
print(f"Feature matrix shape: {X_cpu.shape}")

# ─── MOVE TO GPU ──────────────────────────────────────────────────────────────

print("Moving data to GPU...")
X = np.array(X_cpu)
y = np.array(y_cpu)
print("Data on GPU — starting training...")

# ─── TRAIN ────────────────────────────────────────────────────────────────────

w, b = train(X, y, learning_rate=0.1, iterations=5000, log_every=500)

# ─── SAVE MODEL ───────────────────────────────────────────────────────────────

model = {
    "vocabulary": vocabulary,
    "weights": np.asnumpy(w).tolist(),
    "bias": float(b)
}

with open("logistic_model.json", "w", encoding="utf-8") as f:
    json.dump(model, f, ensure_ascii=False, indent=2)

print("Training complete. Model saved as logistic_model.json")