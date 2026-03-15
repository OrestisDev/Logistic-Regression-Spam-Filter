import re
import os
import json
import math
import html
import numpy as np
from html.parser import HTMLParser

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

# ─── LOAD MODEL ───────────────────────────────────────────────────────────────

_model_path = os.path.join(os.path.dirname(__file__), "logistic_model.json")

with open(_model_path, "r", encoding="utf-8") as f:
    _model = json.load(f)

vocabulary = _model["vocabulary"]
word_to_idx = {word: idx for idx, word in enumerate(vocabulary)}
w = np.array(_model["weights"], dtype=np.float32)
b = float(_model["bias"])

# ─── SIGMOID ──────────────────────────────────────────────────────────────────

def sigmoid(z):
    z = np.clip(z, -30, 30)
    return 1 / (1 + np.exp(-z))

# ─── PREDICT ──────────────────────────────────────────────────────────────────

def predict_email(email_text):
    tokens = tokenize_email(email_text)
    x = np.zeros(len(vocabulary), dtype=np.float32)
    for token in tokens:
        if token in word_to_idx:
            x[word_to_idx[token]] += 1
    z = float(np.dot(w, x)) + b
    confidence = float(sigmoid(z))
    label = "spam" if confidence >= 0.5 else "ham"
    return label, round(confidence * 100, 1)
