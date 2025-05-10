# ========== app.py ==========

import streamlit as st
import requests
import joblib
import re
from collections import Counter

# ========== إعداد التوكنات ==========
HUGGINGFACE_API_TOKEN = "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"
GROQ_API_KEY = "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"

# ========== تحميل النماذج المدربة ==========
tfidf = joblib.load('tfidf_vectorizer.pkl')
label_encoder = joblib.load('label_encoder.pkl')
svm_model = joblib.load('svm_model.pkl')

# ========== دوال مساعدة ==========

# تنظيف النص
def clean_text(text):
    text = re.sub(r'[^\u0600-\u06FF\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# التنبؤ بأعلى 3 تصنيفات
def predict_top3(text):
    cleaned = clean_text(text)
    vectorized = tfidf.transform([cleaned])

    try:
        probabilities = svm_model.decision_function(vectorized)
    except:
        probabilities = svm_model.predict_proba(vectorized)

    if len(probabilities.shape) == 1:
        probabilities = probabilities.reshape(1, -1)

    top3_idx = probabilities.argsort()[0][-3:][::-1]
    top3_scores = probabilities[0][top3_idx]
    top3_labels = label_encoder.inverse_transform(top3_idx)

    normalized_scores = (top3_scores - top3_scores.min()) / (top3_scores.max() - top3_scores.min() + 1e-6)
    percentages = (normalized_scores * 100).astype(int)

    return list(zip(top3_labels, percentages))

# تحليل النص (عدد الكلمات + الكلمات المتكررة)
def analyze_text(text):
    words = clean_text(text).split()
    num_words = len(words)
    most_common = Counter(words).most_common(5)
    return num_words, most_common

# تلخيص المقال مع عرض خطأ واضح
def summarize_text(text):
    API_URL = "https://api-inference.huggingface.co/models/sshleifer/distilbart-cnn-12-6"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"}
    payload = {"inputs": text}
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()[0]['summary_text']
    else:
        return f"❌ خطأ تلخيص: {response.status_code} - {response.text}"

# اقتراح عنوان مع عرض خطأ واضح
def suggest_title(text):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "mistral-7b-instruct",
        "messages": [
            {"role": "system", "content": "أنت مساعد ذكي مختص في كتابة عناوين إخبارية جذابة باللغة العربية."},
            {"role": "user", "content": f"اقترح عنوانًا قصيرًا لهذا المقال:\n\n{text}"}
        ],
        "temperature": 0.7,
        "max_tokens": 50
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"].strip()
    else:
        return f"❌ خطأ عنوان: {response.status_code} - {response.text}"

# صفحة حول المشروع
def show_about():
    st.markdown("""
    ## حول المشروع 🧠
    هذا النظام يقوم بتصنيف المقالات الإخبارية المكتوبة باللغة العربية إلى فئات محددة باستخدام الذكاء الاصطناعي.
    
    - **تصنيف المقال** باستخدام SVM.
    - **تلخيص المقال الذكي** باستخدام Huggingface BART.
    - **اقتراح عنوان ذكي** باستخدام Groq Mixtral.
    
    ### إعداد الطالب:
    مشروع لمقرر EMAI 631 – معالجة اللغة الطبيعية (NLP).
    """)

# ========== إعداد واجهة Streamlit ==========

st.set_page_config(
    page_title="تصنيف الأخبار العربية باستخدام الذكاء الاصطناعي",
    page_icon="📰",
    layout="centered"
)

# واجهة RTL
st.markdown(
    """
    <style>
    body, .stApp {
        direction: rtl;
        text-align: right;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📰 تصنيف الأخبار العربية باستخدام الذكاء الاصطناعي")

tabs = st.tabs(["📰 تصنيف وتحليل مقال", "ℹ️ حول المشروع"])

# ======== التبويب الأول: تصنيف وتحليل مقال ========
with tabs[0]:
    st.subheader("📄 أدخل نص المقال:")
    user_input = st.text_area("✍️ اكتب أو الصق نص المقال هنا:", height=250)

    if st.button("🔎 تحليل المقال"):
        if not user_input.strip():
            st.warning("⚠️ الرجاء إدخال نص قبل التصنيف.")
        else:
            # تصنيف
            top3_predictions = predict_top3(user_input)
            
            st.success("✅ أعلى 3 تصنيفات محتملة:")
            for label, percent in top3_predictions:
                st.write(f"🔹 {label}: {percent}%")

            # تحليل إضافي
            st.markdown("---")
            st.info("📊 تحليل نص المقال:")
            num_words, common_words = analyze_text(user_input)
            st.write(f"- عدد الكلمات: {num_words}")
            st.write("- أكثر الكلمات تكراراً:")
            for word, count in common_words:
                st.write(f"  • {word} ({count} مرات)")

            # تلخيص المقال
            st.markdown("---")
            st.success("📝 تلخيص المقال:")
            summary = summarize_text(user_input)
            st.write(summary)

            # اقتراح عنوان للمقال
            st.markdown("---")
            st.success("📰 اقتراح عنوان للمقال:")
            title = suggest_title(user_input)
            st.write(f"**{title}**")

# ======== التبويب الثاني: حول المشروع ========
with tabs[1]:
    show_about()

# ======== تذييل ========
st.markdown("---")
st.caption("🚀 مشروع طلابي مقدم لمقرر EMAI 631 - جميع الحقوق محفوظة © 2025")
