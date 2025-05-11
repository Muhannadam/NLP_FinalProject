# 📰 NLP Final Project - Arabic News Classification and Summarization using AI

Welcome to our final project for **EMAI 631: Natural Language Processing**.

This project demonstrates an integrated NLP system capable of:
- Classifying Arabic news articles into predefined categories.
- Summarizing article content intelligently.
- Suggesting a smart and concise Arabic headline.

The project relies on **Allam** model for generative tasks **only** (summarization and title suggestion).  

---

## 📚 Project Description

We developed an end-to-end Arabic news classification and summarization system by combining classical machine learning (SVM) for classification and generative AI for text summarization and headline generation.

The system was designed to handle Arabic language text properly (Right-to-Left) and deliver fast, secure, and high-quality outputs.

---

## ✨ Key Features

- ✅ Arabic news article classification into 7 major categories using a trained SVM model.
- ✅ Displaying the Top-3 predicted categories with associated confidence percentages.
- ✅ Smart summarization of the article's main content using **Allam**.
- ✅ Generating a concise and relevant Arabic headline for the article.
- ✅ Full Arabic RTL user interface designed with **Streamlit**.
- ✅ Secure handling of API keys using **Streamlit Secrets Management**.
- ✅ Professional UI/UX and responsive application ready for deployment.

---

## 🔥 Technologies Used

| Category | Technologies |
|:--|:--|
| Programming Language | Python 3.10 |
| Machine Learning | Scikit-learn (SVM Classifier) |
| Web Application | Streamlit |
| API Integration | Requests |
| Generative AI | (Allam Model) |
| Data Processing | Pandas, Regular Expressions |
| Model Saving/Loading | Joblib |
| Cloud Notebook (Training/EDA) | Google Colab |

---

## 📦 Dataset Details

- **Dataset Name:** SANAD Dataset
- **Languages:** Arabic
- **Categories:** Finance, Sports, Medical, Technology, Politics, Religion, Culture
- **Number of Samples:** 45,500 articles
- **Preprocessing Steps:**
  - Removing non-Arabic characters and punctuation.
  - Text normalization.
  - Whitespace cleanup.
- **Train/Test Split:** 80% Training - 20% Testing

---

## 🛠️ System Workflow

1. **User Input:** Paste Arabic news article text.
2. **Text Preprocessing:** Clean and prepare the input.
3. **Classification Module:**
   - TF-IDF vectorization.
   - Prediction via SVM Classifier.
   - Display top-3 category predictions with confidence.
4. **Summarization and Headline Generation:**
   - Send cleaned text to (Allam model) through API.
   - Receive:
     - Summarized content.
     - Suggested Arabic headline.
5. **Display:** Neat Streamlit-based interface in Arabic RTL layout.

---

## 🚀 Live Deployment

- 📒 [Google Colab Notebook](https://colab.research.google.com/drive/1A72NnL_KUVrAcqCNKNe-qT3dPusuXpVT#scrollTo=swbzcLZyKyYz)
- 🌐 [Streamlit Application](https://mhnd-nlp.streamlit.app/)
- 💻 [GitHub Repository](https://github.com/Muhannadam/NLP_FinalProject)

---

## 👨‍💻 Team Members

| Name | Student ID |
|:--|:--|
| Muhannad Almuntashiri | 2503649 |
| Mohammed Talal Mursi | 2503652 |
| Ghaith Omar Alhumaidi | 2503650 |