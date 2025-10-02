# app.py - Elegant Iris Classifier 🌸

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# -------------------- Page Config --------------------
st.set_page_config(
    page_title="🌸 Iris Classifier",
    layout="centered",
    page_icon="🌺",
)

# -------------------- Custom Styling --------------------
st.markdown(
    """
    <style>
    body {
        background: linear-gradient(135deg, #fceabb, #f8b500);
        font-family: 'Segoe UI', sans-serif;
    }
    .stSlider label, .stButton button {
        font-size: 18px !important;
    }
    .stButton button {
        background-color: #f8b500;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        height: 3em;
        width: 100%;
    }
    .stButton button:hover {
        background-color: #d79922;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------- Title --------------------
st.title("🌸 Elegant Iris Flower Classifier 🌸")
st.markdown("### Predict the species of an Iris flower using a trained ML model.")

# -------------------- Load Model --------------------
MODEL_PATH = "iris_model.joblib"

try:
    saved = joblib.load(MODEL_PATH)
    model = saved["model"]
    feature_names = saved["feature_names"]
    target_names = saved["target_names"]
except Exception as e:
    st.error(f"⚠️ Model not found! Run `train_model.py` first. Error: {e}")
    st.stop()

# -------------------- Load Dataset --------------------
iris = load_iris(as_frame=True)
df = iris.frame.copy()

# -------------------- Sliders Input --------------------
st.subheader("🌺 Input Flower Features")
st.write("Move the sliders below to set sepal and petal measurements:")

col1, col2 = st.columns(2)
inputs = []
for i, feature in enumerate(feature_names):
    with col1 if i % 2 == 0 else col2:
        val = st.slider(
            label=feature.replace("_", " ").title(),
            min_value=float(df[feature].min()),
            max_value=float(df[feature].max()),
            value=float(df[feature].mean()),
            step=0.1,
        )
        inputs.append(val)

# -------------------- Prediction --------------------
if st.button("✨ Predict Species"):
    prediction = model.predict([inputs])[0]
    prediction_proba = model.predict_proba([inputs])[0]

    st.success(f"🌸 Predicted Species: **{target_names[prediction]}**")

    # Probability Bar Plot
    proba_df = pd.DataFrame({"Species": target_names, "Probability": prediction_proba})
    fig, ax = plt.subplots()
    sns.barplot(x="Species", y="Probability", data=proba_df, palette="pastel", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title("Prediction Probabilities")
    st.pyplot(fig)
