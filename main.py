import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# Load the data
heart_data = pd.read_csv('heart_disease_data.csv')
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split and model training
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, stratify=Y, random_state=42)

# Use Random Forest for better accuracy
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced')
model.fit(X_train, Y_train)

# Accuracy
training_data_accuracy = accuracy_score(model.predict(X_train), Y_train)
test_data_accuracy = accuracy_score(model.predict(X_test), Y_test)

# Streamlit UI Custom CSS and Layout
st.markdown(
    """
    <style>
    .main { background-color: #FAF9F6; padding: 20px; }
    h1, h2, h3 { color: #00539CFF; font-weight: bold; }
    .sidebar .sidebar-content { background-color: #F3E5AB; }
    .stButton>button { 
        color: white; background-color: #FF6F61; font-size: 20px; padding: 10px 20px; 
        border-radius: 10px; transition: 0.3s; 
    }
    .stButton>button:hover { background-color: #FF4C4C; }
    </style>
    """, unsafe_allow_html=True
)

# App Title with Emoji
st.title("❤️ Heart Disease Prediction App")
st.write("### 🩺 Enter health information below to predict heart disease risk.")

# Sidebar Information with Emoji
with st.sidebar:
    st.header("ℹ️ About This App")
    st.write("This app uses a predictive model to assess the likelihood of heart disease based on your input. By analyzing factors such as age, gender, blood pressure, cholesterol levels and other health parameters, the app can estimate your risk of developing heart disease. It helps users understand their heart health and identify areas where they can make positive lifestyle changes to reduce their risk. The app leverages machine learning algorithms to process user data, providing personalized feedback and actionable recommendations.")
    
    st.subheader("⚙️ Model Performance")
    st.write(f"✔️ Training Accuracy: {training_data_accuracy * 100:.2f}%")
    st.write(f"✔️ Test Accuracy: {test_data_accuracy * 100:.2f}%")
    
    st.subheader("📊 Data Sample")
    st.write(heart_data.head())

    # Feature Importance Visualization Section
    feature_importances = pd.Series(model.feature_importances_, index=heart_data.columns[:-1])
    st.write("### 🔍 Feature Importance")
    st.bar_chart(feature_importances)

    # Optional: Sort features for better visualization
    #sorted_importances = feature_importances.sort_values(ascending=False)
    #st.write("### 📊 Sorted Feature Importance")
    #st.bar_chart(sorted_importances)

    st.markdown(
        """
        #### 📝 **Explanation of Dataset Features**
        - **Age**: The age of the patient in years.
        - **Gender**: Male (1), Female (0).
        - **Chest Pain Type**:
          - 0 = Typical Angina
          - 1 = Atypical Angina
          - 2 = Non-Anginal Pain
          - 3 = Asymptomatic
        - **Resting BP**: Blood pressure in mm Hg.
        - **Cholesterol Level**: Serum cholesterol in mg/dL.
        - **Fasting Blood Sugar**: > 120 mg/dL (1 = Yes, 0 = No).
        - **Resting ECG**:  
          - 0 = Normal  
          - 1 = ST-T wave abnormality  
          - 2 = Left Ventricular Hypertrophy  
        - **Max Heart Rate**: Achieved during exercise.
        - **Exercise-Induced Angina**: 1 = Yes, 0 = No.
        - **Old Peak**: ST depression (severity of ischemia).
        - **Slope**:  
          - 0 = Upsloping  
          - 1 = Flat  
          - 2 = Downsloping  
        - **Fluoroscopy**: No. of blood vessels (0–3).
        - **Thallium Test**:  
          - 0 = Normal  
          - 1 = Fixed Defect  
          - 2 = Reversible Defect  
        """
    )

# Health Information Input Section
st.write("### ✍️ Enter Health Information:")
st.markdown("---")

# User Inputs with Icons and Emojis
age = st.number_input("🎂 Age", step=1, min_value=0, max_value=120, value=50)
gender = st.radio("👩‍⚕️ Gender", options=["Male", "Female"])
chest_pain = st.radio("💓 Chest Pain Type", options=["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
resting_bp = st.number_input("🩺 Resting Blood Pressure (mm Hg)", step=1, min_value=0, max_value=300, value=120)
cholesterol = st.number_input("🍔 Cholesterol Level (mg/dL)", step=1, min_value=0, max_value=600, value=200)
fasting_blood_sugar = st.radio("🩸 Fasting Blood Sugar > 120 mg/dL", options=["Yes", "No"])
resting_ecg = st.radio("📉 Resting ECG", options=["Normal", "ST-T wave abnormality", "Left Ventricular Hypertrophy"])
max_heart_rate = st.number_input("🏃 Max Heart Rate Achieved", step=1, min_value=0, max_value=250, value=150)
exercise_induced_angina = st.radio("🚴‍♂️ Exercise Induced Angina", options=["Yes", "No"])
old_peak = st.number_input("📊 Old Peak (ST Depression)", step=0.1, min_value=0.0, max_value=10.0, value=1.0)
slope = st.radio("⛰️ Slope of Peak Exercise ST Segment", options=["Upsloping", "Flat", "Downsloping"])
fluoroscopy = st.selectbox("🔬 Fluoroscopy Result (No. of major vessels)", options=[0, 1, 2, 3])
thallium_stress = st.radio("🧪 Thallium Stress Test Result", options=["Normal", "Fixed Defect", "Reversible Defect"])

# Display an image for better visuals
st.image("heartImage.png", use_container_width=True)

# Convert user input to model-friendly format
user_data = np.asarray([
    age,
    1 if gender == "Male" else 0,
    ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"].index(chest_pain),
    resting_bp,
    cholesterol,
    1 if fasting_blood_sugar == "Yes" else 0,
    ["Normal", "ST-T wave abnormality", "Left Ventricular Hypertrophy"].index(resting_ecg),
    max_heart_rate,
    1 if exercise_induced_angina == "Yes" else 0,
    old_peak,
    ["Upsloping", "Flat", "Downsloping"].index(slope),
    fluoroscopy,
    ["Normal", "Fixed Defect", "Reversible Defect"].index(thallium_stress)
]).reshape(1, -1)

user_data_scaled = scaler.transform(user_data)

# Prediction Section with Emoji
if st.button("🔍 Predict"):
    prediction = model.predict(user_data_scaled)
    prediction_proba = model.predict_proba(user_data_scaled)

    result_text = "🚫 Low Risk: This person is unlikely to have heart disease." if prediction[0] == 0 else "⚠️ High Risk: This person is likely to have heart disease."
    result_color = "#4CAF50" if prediction[0] == 0 else "#FF4C4C"
    
    st.markdown(
        f"<h2 style='text-align: center; color: {result_color};'>{result_text}</h2>", 
        unsafe_allow_html=True
    )

    # Visualize Prediction Probabilities
    fig, ax = plt.subplots()
    labels = ["No Heart Disease", "Heart Disease"]
    ax.pie(prediction_proba[0], labels=labels, autopct='%1.1f%%', startangle=90, colors=["#4CAF50", "#FF4C4C"])
    ax.axis("equal")
    st.write("### 📊 Probability of Heart Disease")
    st.pyplot(fig)

# Footer with Emoji
st.write("---")
st.markdown("### 💡 Learn More About Heart Disease")
st.markdown(
    """
    - **What is Heart Disease?** ❤️ Heart disease includes various heart conditions. The most common is coronary artery disease.
    - **Prevention Tips**: 🏃‍♂️ Regular exercise, a balanced diet, and regular health screenings can reduce the risk.
    - **Further Reading**:
        - [American Heart Association](https://www.heart.org/)
        - [World Health Organization - Cardiovascular Diseases](https://www.who.int/health-topics/cardiovascular-diseases)
    """
)

# Disclaimer Section
st.markdown(
    """
    ---
    #### 🛑 Disclaimer
    - This app is for **educational purposes only** and is **not a substitute for professional medical advice, diagnosis, or treatment**.
    - If you have concerns about your heart health, please consult a healthcare professional.
    ---
    """
)

# Footer Section
st.markdown(
    "<footer style='text-align: center; font-size: 14px; color: gray;'>"
    "Developed by **ASRIN TOPAL**. 🛠️  Powered by **Streamlit**, **Scikit-learn**, and **Matplotlib**."
    "<br> Data Source: Kaggle's Heart Disease Dataset."
    "</footer>",
    unsafe_allow_html=True
)