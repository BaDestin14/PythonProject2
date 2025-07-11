# Créer une application streamlit (localement)
#%%writefile app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model and the list of features
# Assuming the model and features list are saved as 'model.pkl' and 'features.pkl'
# You would need to save these from your training script
# For now, we'll create dummy ones for demonstration
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('features.pkl', 'rb') as f:
        features = pickle.load(f)
except FileNotFoundError:
    st.error("Model or features file not found. Please train the model and save 'model.pkl' and 'features.pkl'.")
    st.stop()


st.title('Financial Inclusion Prediction')

st.write("""
Enter the details below to predict if a person has a bank account.
""")

# Create input fields for features
input_data = {}
for feature in features:
    if feature in ['year', 'household_size', 'age_of_respondent']:
        # Assuming these are numerical
        input_data[feature] = st.number_input(f'Enter {feature.replace("_", " ")}', value=0.0)
    elif feature.startswith('country_'):
        # Assuming these are one-hot encoded country features
        input_data[feature] = st.selectbox(f'Select {feature.replace("country_", "").replace("_", " ")}', [True, False])
    elif feature.startswith('uniqueid_'):
         # Assuming uniqueid is not relevant for prediction and should be handled differently
        pass # Skip uniqueid features as they are likely identifiers
    elif feature.startswith('location_type_'):
        input_data[feature] = st.selectbox(f'Select {feature.replace("location_type_", "").replace("_", " ")}', [True, False])
    elif feature.startswith('cellphone_access_'):
        input_data[feature] = st.selectbox(f'Select {feature.replace("cellphone_access_", "").replace("_", " ")}', [True, False])
    elif feature.startswith('gender_of_respondent_'):
        input_data[feature] = st.selectbox(f'Select {feature.replace("gender_of_respondent_", "").replace("_", " ")}', [True, False])
    elif feature.startswith('relationship_with_head_'):
        input_data[feature] = st.selectbox(f'Select {feature.replace("relationship_with_head_", "").replace("_", " ")}', [True, False])
    elif feature.startswith('marital_status_'):
         input_data[feature] = st.selectbox(f'Select {feature.replace("marital_status_", "").replace("_", " ")}', [True, False])
    elif feature.startswith('education_level_'):
         input_data[feature] = st.selectbox(f'Select {feature.replace("education_level_", "").replace("_", " ")}', [True, False])
    elif feature.startswith('job_type_'):
         input_data[feature] = st.selectbox(f'Select {feature.replace("job_type_", "").replace("_", " ")}', [True, False])
    else:
        # Handle any other potential features
        input_data[feature] = st.text_input(f'Enter {feature.replace("_", " ")}')


if st.button('Predict'):
    # Create a DataFrame from input data
    input_df = pd.DataFrame([input_data])

    # Ensure the order of columns matches the training data
    input_df = input_df.reindex(columns=features, fill_value=False)

    # Make prediction
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    st.subheader('Prediction Result:')
    if prediction[0]:
        st.success('The person is likely to have a bank account.')
    else:
        st.error('The person is unlikely to have a bank account.')

    st.subheader('Prediction Probability:')
    st.write(f'Probability of not having a bank account: {prediction_proba[0][0]:.4f}')
    st.write(f'Probability of having a bank account: {prediction_proba[0][1]:.4f}')