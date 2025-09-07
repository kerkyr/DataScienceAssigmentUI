import streamlit as st
import pandas as pd
from joblib import load
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import zipfile
import os

# -------------------- Page Config --------------------
st.set_page_config(
    page_title="AuraCare | Heart Disease Predictor",
    page_icon="❤️"
)

st.title("👨🏻‍⚕️ AuraCare Heart Disease Prediction")
st.markdown(
    """
    **Welcome to the AuraCare Heart Disease Predictor.**  
    Enter your details below to check your heart disease risk instantly.
    """
)

# -------------------- Load Model --------------------
zip_path = "best_rf_model.zip"
unzip_dir = "model/"

# Create directory if not exists
if not os.path.exists(unzip_dir):
    os.makedirs(unzip_dir)

# Extract model if not already extracted
if not os.path.exists(os.path.join(unzip_dir, "best_rf_model.joblib")):
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(unzip_dir)
    except Exception as e:
        st.error(f"Error extracting model: {e}")
        st.stop()

# Load model
try:
    model = load(os.path.join(unzip_dir, "best_rf_model.joblib"))
except FileNotFoundError:
    st.error("❌ Could not find `best_rf_model.joblib` after extraction.")
    st.stop()

# -------------------- Feature List --------------------
expected = list(getattr(model, "feature_names_in_", []))
if not expected:  # fallback if model has no feature_names_in_
    expected = [
        "Fasting Blood Sugar", "BMI", "Cholesterol Level", "Sleep Hours", "Age",
        "Stress Level", "Sugar Consumption", "Exercise Habits",
        "Gender_Male", "Smoking_Yes", "High Blood Pressure_Yes"
    ]

# -------------------- Input Form --------------------
with st.form("predict_form"):
    st.subheader("📋 Personal Health Details")

    age = st.number_input("Age", min_value=0, max_value=100, value=30)
    gender = st.selectbox("Gender", ["Female", "Male"])
    sleep_hours = st.number_input("Sleep Hours", min_value=0.0, max_value=24.0, value=7.0)
    stress_level = st.selectbox("Stress Level", ["Low", "Medium", "High"])
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=22.0)
    cholesterol_level = st.number_input("Total Cholesterol (mg/dL)", min_value=0.0, max_value=450.0, value=190.0)
    fasting_bs = st.selectbox("Fasting Blood Sugar", ["Normal", "High"])
    high_bp = st.selectbox("High Blood Pressure (diagnosed)", ["No", "Yes"])
    exercise_habits = st.selectbox("Exercise Habits", ["Low", "Medium", "High"])
    sugar_consumption = st.selectbox("Sugar Consumption", ["Low", "Medium", "High"])
    smoking = st.selectbox("Smoking", ["No", "Yes"])

    submitted = st.form_submit_button("🔍 Predict")

# -------------------- Prediction --------------------
if submitted:
    label_map = {"Low": 0, "Medium": 1, "High": 2}

    # Create feature vector
    row = {
        "Fasting Blood Sugar": 1 if fasting_bs == "High" else 0,
        "BMI": float(bmi),
        "Cholesterol Level": float(cholesterol_level),
        "Sleep Hours": float(sleep_hours),
        "Age": int(age),
        "Stress Level": label_map[stress_level],
        "Sugar Consumption": label_map[sugar_consumption],
        "Exercise Habits": label_map[exercise_habits],
        "Gender_Male": 1 if gender == "Male" else 0,
        "Smoking_Yes": 1 if smoking == "Yes" else 0,
        "High Blood Pressure_Yes": 1 if high_bp == "Yes" else 0,
    }
    input_df = pd.DataFrame([row], columns=expected)

    with st.expander("🔎 See the exact feature vector sent to the model"):
        st.write(input_df)

    # Run prediction
    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0, 1]

    # -------------------- Result --------------------
    result_box = st.container()
    with result_box:
        st.success(f"🧾 Prediction: **{'HAS Heart Disease' if pred == 1 else 'NO Heart Disease'}**")
        st.metric("Risk Probability", f"{proba:.3f}")

        # Personalized feedback
        if proba < 0.25:
            st.info("✅ Very low risk. Your heart health looks good based on the provided data. Keep maintaining a balanced diet, regular exercise, and sufficient sleep to stay healthy.")
        elif proba < 0.50:
            st.warning("⚠️ Moderate risk. You may have some risk factors for heart disease. Consider monitoring cholesterol, blood sugar, and stress levels, and make small improvements in daily habits to reduce risk.")
        elif proba < 0.75:
            st.error("🚨 High risk. Your predicted risk of heart disease is significant. It’s strongly recommended to consult a healthcare professional for further check-ups and guidance on prevention strategies.")
        else:
            st.error("❗ Very high risk. Your heart health may be in serious danger. Please seek medical advice as soon as possible, as early detection and treatment can significantly improve outcomes.")


    # -------------------- Probability Gauge --------------------
    with st.expander("📊 Probability of Heart Disease", expanded=True):
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=proba,
            title={"text": "Risk Probability", "font": {"size": 16}},
            gauge={
                "axis": {"range": [0, 1]},
                "bar": {"color": "crimson" if proba >= 0.5 else "limegreen"},
                "steps": [
                    {"range": [0, 0.5], "color": "rgba(50,205,50,0.4)"},
                    {"range": [0.5, 1], "color": "rgba(220,20,60,0.4)"}
                ],
                "threshold": {"line": {"color": "white", "width": 3}, "value": proba}
            }
        ))

        fig_gauge.update_layout(width=350, height=250, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_gauge, use_container_width=False)

    # -------------------- Feature Importances --------------------
    with st.expander("📌 Top Feature Importances", expanded=True):
        importances = getattr(model, "feature_importances_", None)
        if importances is not None:
            fi = pd.Series(importances, index=expected).sort_values(ascending=True)
            topk = min(8, len(fi))

            fig_fi, ax_fi = plt.subplots(figsize=(4.5, 2.5))
            fi.tail(topk).plot(kind="barh", ax=ax_fi, color="skyblue")
            ax_fi.set_xlabel("Importance", fontsize=9)
            ax_fi.set_title("Top Feature Importances", fontsize=10)
            fig_fi.tight_layout()
            st.pyplot(fig_fi, use_container_width=False)
        else:
            st.info("This model does not provide feature importances.")
