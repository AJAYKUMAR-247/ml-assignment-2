import os
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from metrics_utils import compute_metrics

st.set_page_config(page_title="Spam Detector - Model Comparison", layout="wide")
BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}
TARGET = "is_spam"


@st.cache_resource
def load_model(name):
    return joblib.load(os.path.join(BASE, "model", MODEL_FILES[name]))


st.title("Email Spam Classification: 5-Model Comparison")
st.caption("UCI Spambase | models are pre-trained and loaded from disk; uploaded data is only used for evaluation.")

with st.sidebar:
    st.header("Controls")
    upload = st.file_uploader("Upload test CSV (features + 'is_spam' column)", type="csv")
    choice = st.selectbox("Select model", list(MODEL_FILES))
    st.markdown("No file? The bundled `test_data.csv` is used.")

df = pd.read_csv(upload) if upload else pd.read_csv(os.path.join(BASE, "test_data.csv"))
st.write(f"**Evaluating on:** {'uploaded file' if upload else 'bundled test_data.csv'} - {df.shape[0]} rows, {df.shape[1]} columns")
with st.expander("Preview data"):
    st.dataframe(df.head(20))

if TARGET not in df.columns:
    st.error(f"Uploaded CSV must contain the target column '{TARGET}'.")
    st.stop()

X, y = df.drop(columns=TARGET), df[TARGET]
model = load_model(choice)
try:
    X = X[list(model.feature_names_in_)]
except KeyError as e:
    st.error(f"Missing required feature columns: {e}")
    st.stop()

m = compute_metrics(model, X, y)
st.subheader(f"Evaluation metrics - {choice}")
cols = st.columns(6)
for c, (k, v) in zip(cols, m.items()):
    c.metric(k, f"{v:.4f}")

left, right = st.columns(2)
with left:
    st.subheader("Confusion matrix")
    fig, ax = plt.subplots(figsize=(4, 3.5))
    sns.heatmap(confusion_matrix(y, model.predict(X)), annot=True, fmt="d", cmap="Blues",
                xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"], ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    st.pyplot(fig)
with right:
    st.subheader("Classification report")
    rep = classification_report(y, model.predict(X), target_names=["Ham", "Spam"], output_dict=True)
    st.dataframe(pd.DataFrame(rep).T.round(3))

st.subheader("All models on this data")
st.dataframe(pd.DataFrame([{"ML Model Name": n, **compute_metrics(load_model(n), X, y)}
                           for n in MODEL_FILES]).set_index("ML Model Name").round(4))
