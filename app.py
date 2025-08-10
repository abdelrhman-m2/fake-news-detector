import streamlit as st
import pandas as pd
import numpy as np
import re
import string
from io import BytesIO
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# تحميل الموديل و الـ vectorizer من ملف model_data.py
from model_data import model, vectorizer

# ---------------------------
# Utility functions
# ---------------------------

def clean_text(text: str) -> str:
    if pd.isnull(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>+", "", text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\w*\d\w*", "", text)
    return text.strip()


def predict_texts(texts, model, vectorizer):
    clean_texts = [clean_text(t) for t in texts]
    X = vectorizer.transform(clean_texts)
    preds = model.predict(X)
    probs = model.predict_proba(X) if hasattr(model, "predict_proba") else None
    return preds, probs


def prepare_download_df(orig_df, preds, probs):
    out = orig_df.copy()
    out['prediction'] = ['True' if p==1 else 'Fake' for p in preds]
    if probs is not None:
        out['prob_true'] = probs[:, 1]
        out['prob_fake'] = probs[:, 0]
    return out


def get_top_features(vectorizer, model, top_n=20):
    try:
        feature_names = np.array(vectorizer.get_feature_names_out())
        coefs = model.coef_[0]
        top_pos_idx = np.argsort(coefs)[-top_n:][::-1]
        top_neg_idx = np.argsort(coefs)[:top_n]
        top_pos = list(zip(feature_names[top_pos_idx], coefs[top_pos_idx]))
        top_neg = list(zip(feature_names[top_neg_idx], coefs[top_neg_idx]))
        return top_pos, top_neg
    except Exception:
        return [], []

# ---------------------------
# Streamlit App
# ---------------------------

st.set_page_config(page_title="Fake News Detector — Modern", layout="wide", initial_sidebar_state="expanded")

# Header
st.markdown("""
<style>
.big-title{font-size:36px; font-weight:700;}
.subtitle{color: #6c757d; margin-bottom: 20px}
.card{background: linear-gradient(90deg, #ffffff 0%, #f7fbff 100%); padding: 18px; border-radius: 12px; box-shadow: 0 6px 20px rgba(46,61,73,0.08);}
</style>
""", unsafe_allow_html=True)

col1, col2 = st.columns([3,1])
with col1:
    st.markdown('<div class="big-title">Fake News Detector</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Upload a CSV, paste a headline, or test sample news — powered by your saved model.</div>', unsafe_allow_html=True)
with col2:
    st.image("https://static.vecteezy.com/system/resources/previews/012/345/678/original/news-headline-modern-flat-illustration-vector.jpg", width=140)

st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("Settings")
    sample_button = st.button("Load demo samples")
    st.write("\n")
    enable_download = st.checkbox("Enable CSV download of predictions", value=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["Single News", "Batch CSV", "Model Insight"])

with tab1:
    st.subheader("Test a single headline/article")
    input_title = st.text_input("Title")
    input_text = st.text_area("Article text (optional)")

    if st.button("Predict single"):
        full_text = (input_title + " " + input_text).strip()
        if not full_text:
            st.warning("Please enter a title or text to predict.")
        else:
            preds, probs = predict_texts([full_text], model, vectorizer)
            label = "True" if preds[0] == 1 else "Fake"
            st.metric("Prediction", label)
            if probs is not None:
                st.write(f"Confidence (True): {probs[0][1]:.3f} — (Fake): {probs[0][0]:.3f}")
            wc = WordCloud(width=600, height=300).generate(clean_text(full_text))
            fig, ax = plt.subplots(figsize=(8,3))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)

with tab2:
    st.subheader("Upload CSV with columns 'title' and/or 'text'")
    uploaded = st.file_uploader("Upload CSV file for batch prediction", type=['csv'])


    if uploaded is not None:
        df = pd.read_csv(uploaded)
        if 'title' in df.columns and 'text' not in df.columns:
            df['text'] = df['title']
        elif 'text' in df.columns and 'title' not in df.columns:
            df['title'] = ''
        elif 'title' not in df.columns and 'text' not in df.columns:
            st.error("CSV must contain at least one of 'title' or 'text' columns.")
            st.stop()

        df['combined'] = (df['title'].fillna('') + ' ' + df['text'].fillna('')).apply(clean_text)
        preds, probs = predict_texts(df['combined'].tolist(), model, vectorizer)
        out_df = prepare_download_df(df, preds, probs)
        st.success("Batch prediction finished")
        st.dataframe(out_df.head(50))

        if enable_download:
            csv = out_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download predictions CSV", data=csv, file_name='predictions.csv', mime='text/csv')

with tab3:
    st.subheader("Model insights & top features")
    top_pos, top_neg = get_top_features(vectorizer, model, top_n=20)
    colp, coln = st.columns(2)
    with colp:
        st.markdown("**Top words pushing prediction to TRUE**")
        st.table(pd.DataFrame(top_pos, columns=['feature', 'coef']))
    with coln:
        st.markdown("**Top words pushing prediction to FAKE**")
        st.table(pd.DataFrame(top_neg, columns=['feature', 'coef']))

    st.markdown("---")
    st.write("If you want to retrain or fine-tune the model, export your labeled CSV (use column 'label' with 0=Fake,1=True) and train offline.")

# Footer
st.markdown('\n---\n*Built with ❤️ — Streamlit*')
