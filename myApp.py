import streamlit as st
import numpy as np
import joblib
import warnings

# Load the trained model
model_path = 'random_forest_model.pkl'  # Update with your model path
model = joblib.load(model_path)


# Function to preprocess input and predict heart disease
def predict_heart_disease(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
    # Example: Encode categorical variables and scale numerical ones
    sex_value = 1 if sex == 'Male' else 0

    # Prepare input data for prediction
    input_data = np.array(
        [age, sex_value, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]).reshape(1, -1)

    # Suppress warning about feature names
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        prediction = model.predict(input_data)

    return prediction


# Main function to run the web application
def main():
    st.title('Heart Disease Prediction')
    st.write('Enter patient details to predict if they have heart disease.')

    # Input fields for user to enter patient details
    age = st.slider('Age', 20, 100, 50)
    sex = st.selectbox('Sex', ['Male', 'Female'])
    cp = st.selectbox('Chest Pain Type', [1, 2, 3, 4])
    trestbps = st.slider('Resting Blood Pressure (mm Hg)', 90, 200, 120)
    chol = st.slider('Serum Cholesterol (mg/dl)', 100, 600, 200)
    fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', [0, 1])
    restecg = st.selectbox('Resting ECG Result', [0, 1, 2])
    thalach = st.slider('Maximum Heart Rate Achieved', 60, 220, 150)
    exang = st.selectbox('Exercise Induced Angina', [0, 1])
    oldpeak = st.slider('ST Depression Induced by Exercise', 0.0, 6.2, 0.0)
    slope = st.selectbox('Slope of the Peak Exercise ST Segment', [0, 1, 2])
    ca = st.selectbox('Number of Major Vessels Colored by Fluoroscopy', [0, 1, 2, 3, 4])
    thal = st.selectbox('Thalassemia Type', [0, 1, 2, 3])

    if st.button('Predict'):
        try:
            sex_value = 1 if sex == 'Male' else 0
            prediction = predict_heart_disease(age, sex_value, cp, trestbps, chol, fbs, restecg, thalach, exang,
                                               oldpeak, slope, ca, thal)

            if prediction[0] == 1:
                st.error('Patient is likely to have heart disease. Further evaluation recommended.')
            else:
                st.success('Patient is unlikely to have heart disease.')

        except Exception as e:
            st.error('An error occurred during prediction. Please check your input and try again.')


if __name__ == '__main__':
    main()
