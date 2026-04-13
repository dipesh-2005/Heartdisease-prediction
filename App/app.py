import streamlit as st
import pandas as pd
import joblib

# Load saved model, scaler, and expected columns
model = joblib.load("Models/heart_disease_model_logistic_regression.pkl")
scaler = joblib.load("Models/heart_disease_scaler.pkl")
expected_columns = joblib.load("Models/heart_disease_features.pkl")

st.title("Heart Stroke Prediction by Dipesh")
st.markdown("Enter the following details to predict the risk of heart stroke:")
# st.write(expected_columns)

# Collect user input
age = st.slider("Age", 18, 100, 40)
sex = st.selectbox("Sex", ["M", "F"])
chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
max_hr = st.slider("Max Heart Rate", 60, 220, 150)
exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# When Predict is clicked
if st.button("Predict"):

    raw_input = pd.DataFrame(0, columns=expected_columns, index=[0])

    raw_input['Age'] = age
    raw_input['RestingBP'] = resting_bp
    raw_input['Cholesterol'] = cholesterol
    raw_input['FastingBS'] = fasting_bs
    raw_input['MaxHR'] = max_hr
    raw_input['Oldpeak'] = oldpeak

    raw_input["Sex_M"] = 1 if sex == "M" else 0
    raw_input['ChestPainType_' + chest_pain] = 1
    raw_input['RestingECG_' + resting_ecg] = 1
    raw_input['ExerciseAngina_' + exercise_angina] = 1
    raw_input['ST_Slope_' + st_slope] = 1
    raw_input = raw_input.reindex(columns=expected_columns, fill_value=0)

    scaled_input = scaler.transform(raw_input)

    prediction = model.predict(scaled_input)[0]

    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease, please consult a doctor immediately!")
    else:
        st.success("✅ Low Risk of Heart Disease, but maintain a healthy lifestyle!")