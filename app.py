%%writefile app.py

# app.py
# debug_app.py
import streamlit as st
import pickle
import pdfplumber
import docx2txt
import re
import io
import numpy as np
from pathlib import Path
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ====== CONFIG ======
MODEL_PATH = "model.pkl"
TFIDF_PATH = "tfidf.pkl"
LE_PATH = "label_encoder.pkl"

# ====== NLTK setup ======
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

st.set_page_config(page_title="Resume Classifier - DEBUG", layout="wide")

# ---------- Utility ----------
@st.cache_data(show_spinner=False)
def load_model_files():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(TFIDF_PATH, "rb") as f:
        tfidf = pickle.load(f)
    with open(LE_PATH, "rb") as f:
        le = pickle.load(f)
    return model, tfidf, le

def extract_text_from_pdf(file_bytes):
    text = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
    return "\n".join(text)

def extract_text_from_docx(file_bytes):
    tmp_path = Path("temp_resume.docx")
    tmp_path.write_bytes(file_bytes)
    text = docx2txt.process(str(tmp_path))
    try:
        tmp_path.unlink()
    except Exception:
        pass
    return text

# IMPORTANT: Use the exact cleaning used during training
def clean_text_training_style(txt: str) -> str:
    if not isinstance(txt, str):
        txt = str(txt)
    txt = re.sub(r"http\S+|www\S+", " ", txt)
    txt = re.sub(r"\S+@\S+", " ", txt)
    txt = re.sub(r"\+?\d[\d\-\s]{7,}\d", " ", txt)
    txt = re.sub(r"[^a-zA-Z]", " ", txt)
    txt = txt.lower()
    tokens = txt.split()
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)

def safe_decode(pred_array, le):
    pred = np.array(pred_array).astype(int)
    return le.inverse_transform(pred)

# ---------- UI ----------
st.title("🧾 Resume Category Classifier — DEBUG MODE")
st.write("This debug build prints intermediate values to help find why predictions become Other/Unknown.")

# upload / example
uploaded_file = st.file_uploader("Upload resume (.pdf or .docx)", type=["pdf", "docx"])
use_example = st.checkbox("Use example text", value=True)
example_text = ("Data scientist with experience in python, machine learning, deep learning, "
                "pandas, scikit-learn, SQL, visualization")

if use_example:
    resume_area = st.text_area("Example/Editable resume text", value=example_text, height=220)
else:
    resume_area = ""

if st.button("Run Prediction (with debug info)"):
    try:
        model, tfidf, le = load_model_files()
    except Exception as e:
        st.error(f"Failed to load model files: {e}")
        st.stop()

    # get raw text
    raw_text = ""
    if uploaded_file is not None:
        b = uploaded_file.read()
        if uploaded_file.name.lower().endswith(".pdf"):
            raw_text = extract_text_from_pdf(b)
        elif uploaded_file.name.lower().endswith(".docx"):
            raw_text = extract_text_from_docx(b)
    elif resume_area:
        raw_text = resume_area

    if not raw_text or raw_text.strip() == "":
        st.error("No text found. Upload a resume or provide example text.")
        st.stop()

    st.subheader("A — Raw / Extracted Text (first 1000 chars)")
    st.code(raw_text[:1000])

    # Clean using training-style cleaner
    cleaned = clean_text_training_style(raw_text)
    st.subheader("B — Cleaned Text (training-style)")
    st.write(cleaned[:1000])
    st.write(f"Tokens count after cleaning: {len(cleaned.split())}")

    # Token / vocab diagnostics
    tokens = cleaned.split()
    tf_vocab = getattr(tfidf, "vocabulary_", None)
    if tf_vocab is None:
        st.warning("TF-IDF object has no 'vocabulary_' attribute. Are you sure this is the trained vectorizer?")
    else:
        # how many tokens intersect vocabulary
        tokens_in_vocab = [t for t in set(tokens) if t in tf_vocab]
        tokens_not_in_vocab = [t for t in set(tokens) if t not in tf_vocab]
        st.subheader("C — Vocabulary Diagnostics")
        st.write(f"TF-IDF vocab size: {len(tf_vocab)}")
        st.write(f"Tokens in cleaned text: {len(set(tokens))}")
        st.write(f"Tokens present in TF-IDF vocab (sample 50): {tokens_in_vocab[:50]}")
        st.write(f"Tokens NOT present in TF-IDF vocab (sample 50): {tokens_not_in_vocab[:50]}")
        st.write("If almost none of your tokens are in the vocabulary, TF-IDF mismatch is the issue.")

    # Vectorize and show sparsity
    try:
        vector = tfidf.transform([cleaned])
    except Exception as e:
        st.error(f"Vectorization error: {e}")
        st.stop()

    nonzeros = vector.nnz if hasattr(vector, "nnz") else np.count_nonzero(vector)
    st.subheader("D — Vector Diagnostics")
    st.write(f"Vector shape: {vector.shape}")
    st.write(f"Non-zero entries (nnz): {nonzeros}")
    st.write(f"Vector sum (approx): {float(vector.sum()):.6f}")
    if nonzeros == 0 or float(vector.sum()) == 0.0:
        st.error("Vector is all zeros — TF-IDF vocabulary does not match cleaned text. This causes Other/Unknown.")

    # Model and label encoder info
    st.subheader("E — Model / LabelEncoder Info")
    try:
        # model classes if available
        model_classes = getattr(model, "classes_", None)
        st.write(f"Model attribute 'classes_': {model_classes}")
    except Exception:
        st.write("Model has no 'classes_' attribute (that's OK for some models).")

    st.write(f"LabelEncoder classes (count {len(le.classes_)}): {list(le.classes_)[:40]}")

    # Prediction
    try:
        raw_pred = model.predict(vector)
    except Exception:
        # try dense fallback
        raw_pred = model.predict(vector.toarray())

    st.subheader("F — Raw Prediction")
    st.write(f"Raw predict output (dtype: {raw_pred.dtype}): {raw_pred}")

    decoded = safe_decode(raw_pred, le)
    st.subheader("G — Decoded Prediction (after label encoding)")
    st.write(decoded)



    # If Other/Unknown, show reasons & suggestions
    if decoded[0] == "Other/Unknown":
        st.error("Decoded prediction is Other/Unknown. Possible reasons below:")
        if nonzeros == 0 or float(vector.sum()) == 0.0:
            st.write("- The TF-IDF vector is all zeros (no tokens matched the TF-IDF vocabulary).")
        else:
            st.write("- The predicted integer label is out of range for label encoder, or the prediction is a string not in encoder classes.")
        st.write("Suggestions:")
        st.write("1. Ensure the TF-IDF loaded here is EXACTLY the same you used during training (pickle the fitted vectorizer).")
        st.write("2. Ensure preprocessing in this app matches the training preprocessing.")
        st.write("3. Check a sample token from your cleaned text and verify it's in tfidf.vocabulary_.")
        st.write("4. Print tfidf.get_feature_names_out() in training environment and compare.")

    st.success("Debugging run completed.")
