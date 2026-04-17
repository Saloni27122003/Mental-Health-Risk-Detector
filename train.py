# ==============================
# 1. Import Libraries
# ==============================
import pandas as pd
import pickle
import re
import nltk

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# ==============================
# 2. Load Dataset
# ==============================
df = pd.read_csv('cleaned_dataset.csv')

# ==============================
# 3. Text Cleaning Function
# ==============================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    
    words = text.split()
    words = [word for word in words if word not in stop_words]
    
    return " ".join(words)

df['text'] = df['text'].apply(clean_text)

# ==============================
# 4. Split Data
# ==============================
X = df['text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# 5. TF-IDF Vectorization
# ==============================
vectorizer = TfidfVectorizer(max_features=5000)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ==============================
# 6. Train Model
# ==============================
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# ==============================
# 7. Evaluate Model
# ==============================
y_pred = model.predict(X_test_vec)

print("\nModel Performance:\n")
print(classification_report(y_test, y_pred))

# ==============================
# 8. Save Model
# ==============================
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))

print("Model and vectorizer saved!")