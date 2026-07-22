import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

st.title("Diabetes Prediction")

# User Inputs
pregnancies = st.number_input("Pregnancies", 0, 20, 1)
glucose = st.number_input("Glucose", 0, 300, 120)
bloodpressure = st.number_input("Blood Pressure", 0, 200, 70)
skinthickness = st.number_input("Skin Thickness", 0, 100, 20)
insulin = st.number_input("Insulin", 0, 900, 80)
bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
age = st.number_input("Age", 1, 100, 30)

if st.button("Predict"):

    sample = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": bloodpressure,
        "SkinThickness": skinthickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age
    }

    sample_df = pd.DataFrame([sample])
    sample_df = sample_df.reindex(columns=columns)
    sample_scaled = scaler.transform(sample_df)

    prediction = model.predict(sample_scaled)

    if prediction[0] == 1:
        st.success("Diabetes: Yes")
    else:
        st.error("Diabetes: No")
