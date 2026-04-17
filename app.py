import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    
    words = text.split()
    words = [word for word in words if word not in stop_words]
    
    return " ".join(words)

st.title("Mental Health Risk Detector")

st.write("⚠️ This is not a medical diagnosis tool.")

user_input = st.text_area("Enter text:")

if st.button("Analyze"):
    cleaned = clean_text(user_input)
    vec = vectorizer.transform([cleaned])
    result = model.predict(vec)
    
    if result[0] == 1:
        st.error("⚠️ Potential Mental Health Risk Detected")
    else:
        st.success("✅ No Significant Risk Detected")