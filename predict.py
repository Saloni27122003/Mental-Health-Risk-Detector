import pickle
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load model
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    
    words = text.split()
    words = [word for word in words if word not in stop_words]
    
    return " ".join(words)

def predict(text):
    text = clean_text(text)
    vec = vectorizer.transform([text])
    result = model.predict(vec)
    
    return "At Risk" if result[0] == 1 else "Not At Risk"

# Test
print(predict("I feel very lonely and hopeless"))