import streamlit as st
import pandas as pd
import joblib
import os
from together import Together

# Load the pre-trained logistic regression model
@st.cache(allow_output_mutation=True)
def load_model():
    return joblib.load('logistic_regression_model.pkl')

model = load_model()

# Streamlit app
st.title("Logistic Regression Class Predictor")

# Input field for Together API key
api_key = st.text_input("Enter your Together API key:")

# File uploader for CSV file
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if st.button("GO"):
    if uploaded_file is not None:
        if api_key:
            # Read the CSV file
            df = pd.read_csv(uploaded_file)

            # Select only numerical columns for prediction
            numerical_columns = df.select_dtypes(include=['number']).columns

            if not numerical_columns.empty:
                # Predict the class for each row in the CSV file
                prediction = model.predict(df[numerical_columns])[0]

                # Convert numeric predictions to class labels
                class_label = 'OK Log' if prediction == 0 else 'Failure'

                # Initialize Together client with the provided API key
                client = Together(api_key=api_key)

                # Generate a user-friendly message
                if class_label == 'OK Log':
                    message_content = "Explain to the user that his log was classified as a safe, good one and everything is great!"
                else:
                    message_content = "Explain to the user that his log was classified as one that is a failure so something did not go well unfortunately"

                response = client.chat.completions.create(
                    model="meta-llama/Llama-3-8b-chat-hf",
                    messages=[{"role": "user", "content": message_content}],
                )

                human_readable_message = response.choices[0].message.content

                # Display the prediction and human-readable message
                st.write(f"Prediction: {class_label}")
                st.write(f"Response: {human_readable_message}")
            else:
                st.write("The uploaded CSV file does not contain any numerical columns for prediction.")
        else:
            st.write("Please enter your Together API key.")
    else:
        st.write("Please upload a CSV file.")
