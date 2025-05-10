# ========== app.py ==========

import streamlit as st
import joblib
import re

# Load models
tfidf = joblib.load('tfidf_vectorizer.pkl')
label_encoder = joblib.load('label_encoder.pkl')
svm_model = joblib.load('svm_model.pkl')

# Function to clean text
def clean_text(text):
    text = re.sub(r'[^\u0600-\u06FF\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Prediction function
def predict_category(text):
    cleaned = clean_text(text)
    vectorized = tfidf.transform([cleaned])
    pred = svm_model.predict(vectorized)
    label = label_encoder.inverse_transform(pred)
    return label[0]

# Streamlit UI
st.title("📰 تصنيف الأخبار العربية باستخدام الذكاء الاصطناعي")
st.write("ادخل مقالاً باللغة العربية وسيقوم النظام بتصنيفه تلقائياً.")

# User Input
user_input = st.text_area("اكتب أو الصق المقال هنا:")

if st.button("تصنيف المقال"):
    if user_input.strip() == "":
        st.warning("⚠️ الرجاء إدخال نص قبل التصنيف.")
    else:
        category = predict_category(user_input)
        st.success(f"✅ التصنيف المتوقع: **{category}**")
