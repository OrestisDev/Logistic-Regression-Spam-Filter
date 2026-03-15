import re
import os
import html
from html.parser import HTMLParser
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# ─── STOPWORDS ────────────────────────────────────────────────────────────────

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

# ─── TOKENIZATION ─────────────────────────────────────────────────────────────

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

def preprocess(text):
    text = text.lower()
    text = strip_html(text)
    return text

# ─── LOAD TRAINING DATA ───────────────────────────────────────────────────────

ham_folder = "logisticdatasets/hams-data"
spam_folder = "logisticdatasets/spams-data"

ham_files = [os.path.join(ham_folder, f) for f in os.listdir(ham_folder) if f.endswith(".txt")]
spam_files = [os.path.join(spam_folder, f) for f in os.listdir(spam_folder) if f.endswith(".txt")]

print("Loading training data...")

ham_emails = []
for file_path in ham_files:
    with open(file_path, "r", encoding="latin-1") as f:
        for line in f:
            line = line.strip()
            if line:
                ham_emails.append(preprocess(line))

spam_emails = []
for file_path in spam_files:
    with open(file_path, "r", encoding="latin-1") as f:
        for line in f:
            line = line.strip()
            if line:
                spam_emails.append(preprocess(line))

print(f"Ham emails:  {len(ham_emails)}")
print(f"Spam emails: {len(spam_emails)}")

train_texts = ham_emails + spam_emails
train_labels = [0] * len(ham_emails) + [1] * len(spam_emails)

# ─── LOAD BENCHMARK DATA ──────────────────────────────────────────────────────

print("Loading benchmark data...")

benchmark_folder = "logisticdatasets/benchmark-testing"
benchmark_files = [
    os.path.join(benchmark_folder, f)
    for f in os.listdir(benchmark_folder)
    if f.endswith(".txt")
]

test_texts = []
test_labels = []

for file_path in benchmark_files:
    with open(file_path, "r", encoding="latin-1") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if i == 0 and line.lower().startswith("email"):
                continue
            parts = re.split(r"\s+", line)
            label = parts[-1]
            if label not in {"0", "1"}:
                raise ValueError(f"Invalid label: {line}")
            email_text = preprocess(" ".join(parts[:-1]))
            test_texts.append(email_text)
            test_labels.append(int(label))

print(f"Benchmark emails: {len(test_texts)}")

# ─── VECTORIZE ────────────────────────────────────────────────────────────────

print("Vectorizing...")
vectorizer = CountVectorizer(
    max_features=20000,
    stop_words=list(STOPWORDS),
    ngram_range=(1, 2)
)

X_train = vectorizer.fit_transform(train_texts)
X_test = vectorizer.transform(test_texts)

# ─── TRAIN SKLEARN ────────────────────────────────────────────────────────────

print("Training sklearn LogisticRegression...")
clf = LogisticRegression(max_iter=1000, solver='lbfgs', C=1.0)
clf.fit(X_train, train_labels)
print("Done.")

# ─── BENCHMARK ────────────────────────────────────────────────────────────────

print("Running benchmark...")
predictions = clf.predict(X_test)

correct = sum(p == t for p, t in zip(predictions, test_labels))
accuracy = correct / len(test_labels) * 100

tp = sum(1 for p, t in zip(predictions, test_labels) if p == 1 and t == 1)
tn = sum(1 for p, t in zip(predictions, test_labels) if p == 0 and t == 0)
fp = sum(1 for p, t in zip(predictions, test_labels) if p == 1 and t == 0)
fn = sum(1 for p, t in zip(predictions, test_labels) if p == 0 and t == 1)

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"\nTotal benchmark emails: {len(test_texts)}")
print(f"Correct predictions:    {correct}")
print(f"Accuracy:               {accuracy:.2f}%")
print("\n--- Confusion Matrix ---")
print(f"TP (Spam detected):     {tp}")
print(f"TN (Ham detected):      {tn}")
print(f"FP (Ham marked spam):   {fp}")
print(f"FN (Spam missed):       {fn}")
print("\n--- Metrics ---")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")