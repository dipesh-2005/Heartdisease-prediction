import streamlit as st
import joblib
import numpy as np
import pandas as pd

model = joblib.load("Models/heart_disease_model_logistic_regression.pkl")
scaler = joblib.load("Models/heart_disease_scaler.pkl")
expected_columns = joblib.load("Models/heart_disease_features.pkl")

st.title('Heart Stroke Prediction By Dipesh')
st.markdown('This app predicts the likelihood of heart disease stroke on user input.')


age = st.slider('Age',1,100,25)
sex = st.selectbox('Sex', ['Male', 'Female'])
chest_pain = st.selectbox('Chest Pain Type', ['Typical Angina', 'Atypical Angina', 'Non-Anginal Pain', 'Asymptomatic'])
resting_bp = st.slider('Resting Blood Pressure (mm Hg)', 80, 200, 120)
cholesterol = st.slider('Cholesterol (mg/dl)', 100, 400, 200)
fasting_bs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ['Yes', 'No'])
resting_ecg = st.selectbox('Resting Electrocardiographic Results', ['Normal',  'ST-T Wave Abnormality', 'Left Ventricular Hypertrophy']) 
max_heart_rate = st.slider('Maximum Heart Rate Achieved', 60, 220, 150)
exercise_angina = st.selectbox('Exercise Induced Angina', ['Yes', 'No'])
oldpeak = st.slider('Oldpeak (ST depression induced by exercise)', 0.0, 10.0, 1.0) 
st_slope = st.selectbox('ST Slope', ['Up', 'Flat', 'Down']) 


if st.button('Predict'):
    input_data = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_heart_rate,
        'Oldpeak': oldpeak,
        'Sex_'+ sex: 1,
        'ChestPainType_'+ chest_pain: 1,
        'RestingECG_'+ resting_ecg: 1,
        'ExerciseAngina_'+ exercise_angina: 1,
        'ST_Slope_'+ st_slope: 1
    }

    input_df = pd.DataFrame([input_data])
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0   
    input_df = input_df[expected_columns]
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]


    if prediction == 1:
        st.error('High risk of heart disease stroke. Please consult a doctor.')
    else:        
        st.success('Low risk of heart disease stroke. Keep up the healthy lifestyle!')          