import streamlit as st
import pickle
import numpy as np
model = pickle.load(open("heart_disease_model.pkl", 'rb'))
scaler = pickle.load(open("scaler.pkl", 'rb'))
st.title("Heart Disease Predictor")

st.header("Enter Patient Details:")
age = st.number_input("Age", min_value=1, max_value=120)
sex = st.selectbox("Sex", ["M", "F"])
chest_pain = st.selectbox("Chest Pain Type", ["TA", "ATA", "NAP", "ASY"])
resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0, max_value=250)
cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=0, max_value=600)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
rest_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
max_hr = st.number_input("Maximum Heart Rate Achieved", min_value=0, max_value=250)
exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
oldpeak = st.number_input("Oldpeak (ST Depression)", min_value=0.0, max_value=10.0, step=0.1)
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

sex = 1 if sex == "M" else 0
exercise_angina = 1 if exercise_angina == "Y" else 0
chest_pain_mapping = {"ASY": 0, "ATA": 1, "NAP": 2, "TA": 3}
rest_ecg_mapping = {"LVH": 0, "Normal": 1, "ST": 2}
st_slope_mapping = {"Down": 0, "Flat": 1, "Up": 2}

input_data = np.array([age,sex,chest_pain_mapping[chest_pain],resting_bp,cholesterol,fasting_bs,rest_ecg_mapping[rest_ecg],max_hr,exercise_angina,oldpeak,st_slope_mapping[st_slope]]).reshape(1, -1)

input_data_scaled = scaler.transform(input_data)

if st.button("Predict"):
    prediction = model.predict(input_data_scaled)[0]
    probability = model.predict_proba(input_data_scaled)[0][1]
    if prediction == 1:
        st.error(f" The model predicts **Heart Disease Risk**.\nEstimated Probability: **{probability*100:.2f}%**")
    else:
        st.success(f" The model predicts **No Heart Disease**.\nEstimated Probability: **{probability*100:.2f}%**")
