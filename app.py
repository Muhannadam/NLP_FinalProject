# ========== app.py ==========

import streamlit as st
import joblib
import re
from collections import Counter
import matplotlib.pyplot as plt

# تحميل النماذج المحفوظة
tfidf = joblib.load('tfidf_vectorizer.pkl')
label_encoder = joblib.load('label_encoder.pkl')
svm_model = joblib.load('svm_model.pkl')

# دالة تنظيف النصوص
def clean_text(text):
    text = re.sub(r'[^\u0600-\u06FF\s]', '', text)  # إبقاء الأحرف العربية فقط
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# دالة التنبؤ بأعلى 3 تصنيفات
def predict_top3(text):
    cleaned = clean_text(text)
    vectorized = tfidf.transform([cleaned])
    probabilities = svm_model.decision_function(vectorized)

    # إذا كان النموذج لا يدعم decision_function (بعض النماذج)، استخدم predict_proba
    if len(probabilities.shape) == 1:
        probabilities = probabilities.reshape(1, -1)

    top3_idx = probabilities.argsort()[0][-3:][::-1]
    top3_probs = probabilities[0][top3_idx]
    top3_labels = label_encoder.inverse_transform(top3_idx)

    # تطبيع القيم إلى نسب مئوية
    normalized_scores = (top3_probs - top3_probs.min()) / (top3_probs.max() - top3_probs.min() + 1e-6)
    percentages = (normalized_scores * 100).astype(int)

    return list(zip(top3_labels, percentages))

# دالة التحليل الإحصائي للنص
def analyze_text(text):
    words = clean_text(text).split()
    num_words = len(words)
    most_common = Counter(words).most_common(5)
    return num_words, most_common

# دالة عرض صفحة حول المشروع
def show_about():
    st.markdown("""
    ## حول المشروع
    هذا المشروع يقدم نظامًا ذكيًا لتصنيف الأخبار العربية إلى فئات محددة باستخدام تقنيات معالجة اللغة الطبيعية (NLP) وخوارزميات تعلم الآلة (SVM).
    
    - **مجموعة البيانات**: SANAD Dataset
    - **النماذج المستخدمة**: TF-IDF + SVM
    - **أعلى 3 احتمالات**: النظام يعرض لك ثلاث احتمالات مع نسبة الثقة لكل فئة
    - **تحليل نصي بسيط**: عرض عدد الكلمات وأكثر الكلمات تكراراً
    """)

# إعداد الصفحة
st.set_page_config(page_title="تصنيف الأخبار العربية", layout="centered", page_icon="📰")

# العنوان الرئيسي
st.title("📰 تصنيف الأخبار العربية باستخدام الذكاء الاصطناعي")

# تبويبات
tabs = st.tabs(["تصنيف مقال", "حول المشروع"])

# التبويب الأول: تصنيف مقال
with tabs[0]:
    st.subheader("ادخل نص المقال أدناه:")
    user_input = st.text_area("✍️ أدخل أو الصق نص المقال هنا:", height=250)

    if st.button("🔎 تصنيف المقال"):
        if user_input.strip() == "":
            st.warning("⚠️ الرجاء إدخال نص قبل التصنيف.")
        else:
            # التنبؤ
            top3_predictions = predict_top3(user_input)
            
            # عرض النتائج
            st.success("✅ أعلى 3 احتمالات للتصنيف:")
            for label, percent in top3_predictions:
                st.write(f"🔹 {label} - {percent}%")

            # تحليل النص
            st.markdown("---")
            st.info("📊 تحليل المقال:")
            num_words, most_common_words = analyze_text(user_input)
            st.write(f"- عدد الكلمات: {num_words}")
            st.write("- أكثر الكلمات تكراراً:")
            for word, count in most_common_words:
                st.write(f"  • {word} ({count} مرات)")

# التبويب الثاني: حول المشروع
with tabs[1]:
    show_about()

# تذييل الصفحة
st.markdown("---")
st.caption("تم تطويره بواسطة الذكاء الاصطناعي 2025 ©")
