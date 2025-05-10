# ========== app.py ==========

import streamlit as st
import joblib
import re
from collections import Counter
import matplotlib.pyplot as plt
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

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

# تلخيص النص
def summarize_text(text, sentence_count=3):
    parser = PlaintextParser.from_string(text, Tokenizer("arabic"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, sentence_count)
    summarized_text = " ".join(str(sentence) for sentence in summary)
    return summarized_text

# صفحة حول المشروع
def show_about():
    st.markdown("""
    ## حول المشروع 🧠
    هذا النظام يقوم بتصنيف المقالات الإخبارية المكتوبة باللغة العربية إلى فئات محددة باستخدام الذكاء الاصطناعي.
    
    - **مجموعة البيانات**: SANAD Dataset.
    - **التمثيل النصي**: TF-IDF Vectorization.
    - **النموذج المستخدم**: Support Vector Machine (SVM).
    - **ميزات إضافية**: عرض أفضل 3 تصنيفات، نسبة الثقة، تحليل نصي، وتلخيص المقال تلقائياً.
    
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

tabs = st.tabs(["📰 تصنيف مقال", "ℹ️ حول المشروع"])

# ======== التبويب الأول: تصنيف مقال ========
with tabs[0]:
    st.subheader("📄 أدخل نص المقال:")
    user_input = st.text_area("✍️ اكتب أو الصق نص المقال هنا:", height=250)

    if st.button("🔎 تصنيف وتلخيص المقال"):
        if not user_input.strip():
            st.warning("⚠️ الرجاء إدخال نص قبل التصنيف.")
        else:
            top3_predictions = predict_top3(user_input)
            
            st.success("✅ أعلى 3 تصنيفات محتملة:")
            for label, percent in top3_predictions:
                st.write(f"🔹 {label}: {percent}%")

            # تحليل إضافي للنص
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
            try:
                summary = summarize_text(user_input, sentence_count=3)
                st.write(summary if summary else "لا يوجد ملخص مناسب للمقال.")
            except Exception as e:
                st.warning("⚠️ تعذر تلخيص النص بسبب قصر المقال أو مشكلة في التحليل.")

# ======== التبويب الثاني: حول المشروع ========
with tabs[1]:
    show_about()

# ======== تذييل ========
st.markdown("---")
st.caption("🚀 مشروع طلابي مقدم لمقرر EMAI 631 - جميع الحقوق محفوظة © 2025")
