import streamlit as st
import joblib

st.title("❤️ Heart Disease Prediction System")

model = joblib.load("Heart_Disease_Model.pkl")

age = st.number_input("Age")

sex = st.selectbox(
    "Sex",
    ["Male", "Female"]
)

if sex == "Male":
    sex = 1
else:
    sex = 0

cp = st.number_input("Chest Pain Type (0-3)")

trestbps = st.number_input("Resting Blood Pressure")

chol = st.number_input("Cholesterol")

fbs = st.selectbox(
    "Fasting Blood Sugar > 120 mg/dl",
    ["No", "Yes"]
)

if fbs == "Yes":
    fbs = 1
else:
    fbs = 0

restecg = st.number_input("Rest ECG (0-2)")

thalach = st.number_input("Maximum Heart Rate")

exang = st.selectbox(
    "Exercise Induced Angina",
    ["No", "Yes"]
)

if exang == "Yes":
    exang = 1
else:
    exang = 0

oldpeak = st.number_input("Oldpeak")

slope = st.number_input("Slope (0-2)")

ca = st.number_input("Number of Major Vessels (0-4)")

thal = st.number_input("Thal (0-3)")

if st.button("Predict"):

    prediction = model.predict([[
        age,
        sex,
        cp,
        trestbps,
        chol,
        fbs,
        restecg,
        thalach,
        exang,
        oldpeak,
        slope,
        ca,
        thal
    ]])

    st.success("Prediction Generated Successfully")

    if prediction[0] == 1:
        st.error("⚠️ Heart Disease Detected")
    else:
        st.success("✅ No Heart Disease Detected")
