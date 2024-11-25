# Heart Disease Prediction App ❤️

This is a web application built using **Streamlit** to predict the likelihood of heart disease based on various health factors. The app leverages machine learning models, specifically a **Random Forest Classifier**, to analyze input data such as age, cholesterol levels, blood pressure, and other vital health metrics to predict the risk of heart disease.

# [HEART DISEASE PREDICTION APP GIVE IT A TRY](https://heart-disease-prediction-app-by-asrin-topal.streamlit.app)

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Information](#model-information)
- [App Screenshot](#app-screenshot)
- [License](#license)

## Overview

The **Heart Disease Prediction App** uses the **Heart Disease Dataset** available on Kaggle to train a machine learning model. Users can input their health information (such as age, gender, cholesterol levels, etc.) into the app to receive a prediction of their heart disease risk. The app also provides features like visualizing the feature importance and the prediction probabilities.

This app helps raise awareness about heart disease and can serve as an educational tool for understanding heart health.

## Features

- **Heart Disease Prediction**: Based on the user's input, the app predicts whether the user has a high or low risk of heart disease.
- **Model Performance**: Displays the model's training and test accuracy.
- **Feature Importance**: Visualizes the importance of each feature in the model's decision-making.
- **Probability Visualization**: Displays the probability of having heart disease using a pie chart.
- **User-Friendly Interface**: Built with Streamlit, the app is interactive and easy to use.

## Installation

### Prerequisites

- Python 3.7 or higher
- Streamlit
- Scikit-learn
- Pandas
- Matplotlib

### Step 1: Clone the Repository

```bash
git clone https://github.com/AsrinTopal/Heart-Disease-Prediction.git
cd Heart-Disease-Prediction
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run the Streamlit App
```
streamlit run main.py
```
#### The app will launch in your browser. If it doesn't open automatically, navigate to http://localhost:8501 in your browser.

## Usage
1. Input Health Information: Enter your personal health information (e.g., age, gender, cholesterol level, etc.) into the provided form on the app's sidebar.

2. Predict Heart Disease Risk: Click the "🔍 Predict" button to get the prediction.

3. View Results: The app will display whether you are at a high or low risk of heart disease, along with the probability distribution.

## Model Information
- Model Used: Random Forest Classifier
- Features: The model uses several health-related features, including age, blood pressure, cholesterol level, chest pain type, and more.
- Training Accuracy: 100%
- Test Accuracy: 81.97% 

## Dataset
The app uses the Heart Disease Dataset available on Kaggle, which includes various health factors and the presence or absence of heart disease.
- Dataset Source: [Kaggle - Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)

## App Screenshot
![heartImage](https://github.com/user-attachments/assets/78265a30-c42a-4b1b-8fb4-04b1c582b54d)
![Untitled-1_enhanced](https://github.com/user-attachments/assets/070f4032-9002-482e-a4d4-ba62a3723088)
![Ekran görüntüsü 2024-11-24 163121](https://github.com/user-attachments/assets/6656c88a-03e3-44b6-8063-c37794730452)
![Ekran görüntüsü 2024-11-24 163427](https://github.com/user-attachments/assets/53489b5c-4e72-4268-a483-ed05949b9351)
![Ekran görüntüsü 2024-11-24 163523](https://github.com/user-attachments/assets/19628a65-7c60-4d8c-a107-2810a2ce26a3)
![Ekran görüntüsü 2024-11-24 163659](https://github.com/user-attachments/assets/edea5fbe-b89f-4a4f-919c-3f440daa6101)


## Explanation of Dataset Features
- Age
- Gender
- Chest Pain Type
- Resting Blood Pressure
- Cholesterol Level
- Fasting Blood Sugar
- Resting ECG
- Max Heart Rate Achieved
- Exercise-Induced Angina
- Old Peak
- Slope of Peak Exercise ST Segment
- Fluoroscopy Result
- Thallium Stress Test Result

## Learn More About Heart Disease
You can learn more about heart disease by visiting the following resources:

- [American Heart Association](https://www.heart.org/)
- [World Health Organization - Cardiovascular Diseases](https://www.who.int/health-topics/cardiovascular-diseases)

## License
This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.
