# ========== app.py ==========

import streamlit as st
import requests
import joblib
import re
import os

# ========== إعداد التوكنات ==========
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# ========== تحميل النماذج المدربة ==========
tfidf = joblib.load('tfidf_vectorizer.pkl')
label_encoder = joblib.load('label_encoder.pkl')
svm_model = joblib.load('svm_model.pkl')
lr_model = joblib.load('lr_model.pkl')


# ========== دوال مساعدة ==========

# تنظيف النص
def clean_text(text):
    text = re.sub(r'[^\u0600-\u06FF\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# التنبؤ بأعلى 3 تصنيفات
def predict_top3_with_model(text, model_name):
    cleaned = clean_text(text)
    vectorized = tfidf.transform([cleaned])

    # اختيار النموذج حسب اسم المستخدم
    if model_name == "Logistic Regression":
        probabilities = lr_model.decision_function(vectorized)
    else:
        probabilities = svm_model.decision_function(vectorized)

    if len(probabilities.shape) == 1:
        probabilities = probabilities.reshape(1, -1)

    top3_idx = probabilities.argsort()[0][-3:][::-1]
    top3_scores = probabilities[0][top3_idx]
    top3_labels = label_encoder.inverse_transform(top3_idx)

    normalized_scores = (top3_scores - top3_scores.min()) / (top3_scores.max() - 1e-6)
    percentages = (normalized_scores * 100).astype(int)

    return list(zip(top3_labels, percentages))


# تلخيص واقتراح عنوان عبر Groq API
def summarize_and_suggest_title(text):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "allam-2-7b",
        "messages": [
            {"role": "system", "content": "أنت مساعد ذكي. عندما يصلك نص طويل، قم بتلخيصه بشكل مختصر، ثم اقترح عنوانًا قصيرًا وجذابًا باللغة العربية."},
            {"role": "user", "content": f"هذا هو نص المقال:\n\n{text}\n\nرجاءً: 1- لخص المقال في فقرة قصيرة. 2- اقترح عنوانًا ذكيًا للمقال."}
        ],
        "temperature": 0.5,
        "max_tokens": 500
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        reply = response.json()["choices"][0]["message"]["content"].strip()
        return reply
    else:
        return f"❌ خطأ في الاتصال: {response.status_code} - {response.text}"

# صفحة حول المشروع
def show_about():
    st.markdown("""
    ## حول المشروع 🧠
    هذا النظام يقوم بتصنيف المقالات الإخبارية المكتوبة باللغة العربية باستخدام الذكاء الاصطناعي.
    
    - **تصنيف المقال** باستخدام SVM.
    - **تلخيص المقال واقتراح عنوان** باستخدام Groq - Allam 2 7B.
    
    ### 
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

    model_choice = st.selectbox(
    "اختر نموذج التصنيف",
    ("SVM", "Logistic Regression")
)

    if st.button("🔎 تحليل المقال"):
        if not user_input.strip():
            st.warning("⚠️ الرجاء إدخال نص قبل التصنيف.")
        else:
            # استدعاء الدالة مع اختيار النموذج
            top3_predictions = predict_top3_with_model(user_input, model_choice)

            st.success("✅ أعلى 3 تصنيفات محتملة:")
            for label, percent in top3_predictions:
                st.write(f"🔹 {label}: {percent}%")

            st.markdown("---")
            st.success("📝 تلخيص المقال واقتراح عنوان:")
            result = summarize_and_suggest_title(user_input)
            st.write(result)


# ======== التبويب الثاني: حول المشروع ========
with tabs[1]:
    show_about()

# ======== تذييل ========
st.markdown("---")
st.caption(" مشروع مقدم لمقرر EMAI 631 - جميع الحقوق محفوظة © 2025")
