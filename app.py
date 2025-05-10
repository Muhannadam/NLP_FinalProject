# ========== مقطع التصنيف بعد التعديل النهائي ==========

if st.button("🔎 تصنيف وتحليل المقال"):
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

        # تفسير القرار
        st.markdown("---")
        st.success("🧠 تفسير قرار التصنيف (أهم الكلمات المؤثرة):")
        important_words = explain_decision(user_input, top_n=5)
        if important_words:
            st.write(", ".join(important_words))
        else:
            st.write("لا توجد كلمات مؤثرة كافية.")
