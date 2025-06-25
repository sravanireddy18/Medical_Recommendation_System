import streamlit as st
import numpy as np
import pandas as pd
import pickle
import base64
import ast

# Load the trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Load additional data
description_df = pd.read_csv(r"description.csv")
diets_df = pd.read_csv(r"diets.csv")
medications_df = pd.read_csv(r"medications.csv")
precautions_df = pd.read_csv(r"precautions_df.csv")
workout_df = pd.read_csv(r"workout_df.csv")
symptom_severity = pd.read_csv(r"Symptom-severity.csv")

# Get the list of symptoms
symptoms_list = symptom_severity["Symptom"].unique()
model_features = model.feature_names_in_

# Initialize session state
if 'predicted_disease' not in st.session_state:
    st.session_state.predicted_disease = None

# Function to set background image
def set_background(local_image_path):
    with open(local_image_path, "rb") as file:
        encoded_image = base64.b64encode(file.read()).decode()
    page_bg = f'''
    <style>
    .stApp {{
        background-image: url("data:image/jpeg;base64,{encoded_image}");
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
    }}
    </style>
    '''

    st.markdown(page_bg, unsafe_allow_html=True)

# Call the function with your image path
set_background(r"Bg.jpg")

# UI: Title and Subtitle
st.markdown("""
    <h1 style='text-align: center; color: white;'>🩺 MedAssist: Your Personal Healthcare Guide</h1>
    <p style='text-align: center; font-size: 20px; font-weight: bold;'>Enter symptoms to predict the disease and get personalized recommendations</p>
""", unsafe_allow_html=True)

# UI: Symptom Selector
selected_symptoms = st.multiselect("Select Symptoms", symptoms_list)

# Prediction Logic
if st.button("🔍 Predict Disease", use_container_width=True):
    if selected_symptoms:
        input_data = np.zeros(len(model_features))
        for symptom in selected_symptoms:
            if symptom in model_features:
                index = np.where(model_features == symptom)[0][0]
                input_data[index] = 1
        
        if len(input_data) != model.n_features_in_:
            st.warning(f"Expected {model.n_features_in_} features, but got {len(input_data)}.")
        else:
            st.session_state.predicted_disease = model.predict([input_data])[0]
    else:
        st.warning("Please select at least one symptom.")

# Results Section
# 🟦 Predicted Disease
if st.session_state.predicted_disease:
    st.markdown(f"""
        <div style='background-color:#e6f0ff; padding:10px; border-radius:10px; text-align:center; border: 1px solid #3399ff;'>
            <h2 style='font-size: 26px;'>🎯 <strong>Predicted Disease: {st.session_state.predicted_disease}</strong></h2>
        </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    # 📜 Description
    with col1:
        if st.button("📜 Show Description", use_container_width=True):
            description = description_df[description_df["Disease"] == st.session_state.predicted_disease]["Description"].values[0]
            st.markdown(f"""
                <div style='background-color:#fffbe6; padding:15px; border-radius:10px; border: 1px solid #ffe58f;'>
                    <p style='font-weight:bold; font-size:19px;'>📌 Description:</p>
                    <p style='font-weight:bold;'>{description}</p>
                </div>
            """, unsafe_allow_html=True)

    # ⚠️ Precautions
    with col2:
        if st.button("⚠️ Show Precautions", use_container_width=True):
            precautions = precautions_df[precautions_df["Disease"] == st.session_state.predicted_disease].iloc[:, 1:].values.flatten()
            precautions = [p for p in precautions if isinstance(p, str) and p.lower() != "nan"]
            precautions_html = "<ul style='font-weight: bold;'>"
            for p in precautions:
                precautions_html += f"<li>{p}</li>"
            precautions_html += "</ul>"
            st.markdown(f"""
                <div style='background-color:#fffbe6; padding:15px; border-radius:10px; border: 1px solid #ffe58f;'>
                    <p style='font-weight:bold; font-size:19px;'>⚠️ Precautions:</p>
                    {precautions_html}
                </div>
            """, unsafe_allow_html=True)

    # 💊 Medications
    with col3:
        if st.button("💊 Show Medications", use_container_width=True):
            meds_data = medications_df[medications_df["Disease"] == st.session_state.predicted_disease]["Medication"].values
            if len(meds_data) > 0:
                try:
                    med_list = ast.literal_eval(meds_data[0])
                    med_list = [m for m in med_list if isinstance(m, str) and m.lower() != "nan"]
                    meds_html = "<ul style='font-weight: bold;'>"
                    for med in med_list:
                        meds_html += f"<li>{med}</li>"
                    meds_html += "</ul>"
                    st.markdown(f"""
                        <div style='background-color:#fffbe6; padding:15px; border-radius:10px; border: 1px solid #ffe58f;'>
                            <p style='font-weight:bold; font-size:19px;'>💊 Medications:</p>
                            {meds_html}
                        </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error parsing medication list: {e}")
            else:
                st.warning("No medication data found.")

    col4, col5 = st.columns(2)

    # 🥗 Diet
    with col4:
        if st.button("🥗 Show Diet Recommendations", use_container_width=True):
            diet_data = diets_df[diets_df["Disease"] == st.session_state.predicted_disease]["Diet"].values
            if len(diet_data) > 0:
                try:
                    # Convert string representation of list to actual list
                    diet_list = ast.literal_eval(diet_data[0])
                    diet_list = [d for d in diet_list if isinstance(d, str) and d.lower() != "nan"]
                    diet_html = "<ul style='font-weight: bold;'>"
                    for d in diet_list:
                        diet_html += f"<li>{d}</li>"
                    diet_html += "</ul>"
                    st.markdown(f"""
                        <div style='background-color:#fffbe6; padding:15px; border-radius:10px; border: 1px solid #ffe58f;'>
                            <p style='font-weight:bold; font-size:19px;'>🥗 Diet Recommendations:</p>
                            {diet_html}
                        </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error parsing diet list: {e}")
            else:
                st.warning("No diet data found.")

        # 🏋️ Workout
    with col5:
        if st.button("🏋️ Show Workout Suggestions", use_container_width=True):
            workout = workout_df[workout_df["disease"] == st.session_state.predicted_disease]["workout"].values
            workout = [w for w in workout if isinstance(w, str) and w.lower() != "nan"]
            workout_html = "<ul style='font-weight: bold;'>"
            for w in workout:
                workout_html += f"<li>{w}</li>"
            workout_html += "</ul>"
            st.markdown(f"""
                <div style='background-color:#fffbe6; padding:15px; border-radius:10px; border: 1px solid #ffe58f;'>
                    <p style='font-weight:bold; font-size:19px;'>🏋️ Workout Suggestions:</p>
                    {workout_html}
                </div>
            """, unsafe_allow_html=True)
