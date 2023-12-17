import streamlit as st
import pickle
import pandas as pd

# Load the trained Random Forest model
with open('rf_model.pkl', 'rb') as model_file:
    loaded_rf_model = pickle.load(model_file)

# Function to get user input for 10 questions
def get_user_input():
    st.title("Child Mental Health Prediction")
    st.write("Answer the following 10 questions to predict the likelihood of mental health issues.")

    questions = [
        "Is the child experiencing changes in sleep patterns?",
        "Does the child show interest in activities they used to enjoy?",
        "Is the child having difficulty concentrating in school?",
        "Has the child exhibited changes in appetite or weight?",
        "Does the child frequently express feelings of sadness or hopelessness?",
        "Is the child having trouble making and maintaining friendships?",
        "Has the child engaged in self-harming behaviors?",
        "Does the child often complain of physical symptoms without a medical cause?",
        "Is there a noticeable decline in academic performance?",
        "Has the child experienced any traumatic events recently?"
    ]

    user_answers = []
    for i, question in enumerate(questions, start=1):
        answer = st.radio(f"Q{i}: {question}", ["Yes", "No"])
        user_answers.append(1 if answer == "Yes" else 0)

    return pd.DataFrame([user_answers], columns=[f'Q{i}' for i in range(1, 11)])

# Get user input
user_input_df = get_user_input()

# Make prediction using the loaded Random Forest model
prediction = loaded_rf_model.predict(user_input_df)[0]

# Map predictions to labels
prediction_label = 'Yes' if prediction == 2 else 'No' if prediction == 1 else 'Maybe'

# Display the prediction
st.subheader("Prediction:")
# st.write(f"The model predicts that the child's mental health status is: {prediction_label}")
mental_health = "has good mental health status" if prediction_label == "No" else "has bad mental health status. So please consult a good psychologist" if prediction_label == "Yes" else "require more diagnosis"
st.write(f"Your child {mental_health}...")

# Optionally, you can display the probability scores for each class
# prediction_probabilities = loaded_rf_model.predict_proba(user_input_df)[0]



# st.subheader("Prediction Probabilities:")
# for class_name, probability in zip(loaded_rf_model.classes_, prediction_probabilities):
#     progress_bar = st.progress(probability)
#     st.write(f"{class_name}: {probability:.2%}")
# To run command: streamlit run app.py
