import streamlit as st

st.title("🩺 About MedAI - Medical Recommendation System")

st.markdown("""
**MedAI** is an intelligent Medical Recommendation System that helps users identify potential diseases based on the symptoms they provide and supports them with complete health guidance.  
This system aims to assist individuals in understanding their health better and taking informed actions through early insights and lifestyle recommendations.

---

### 🔍 What MedAI Offers

- **Disease Prediction**  
  Users can select symptoms they are experiencing, and MedAI predicts the most probable disease using machine learning.

- **Precautionary Measures**  
  Based on the predicted disease, the system provides a list of precautions the user should take to avoid worsening the condition or spreading it (if contagious).

- **Diet Recommendations**  
  Offers suggestions on what to eat and avoid during the illness, helping users plan meals that support faster recovery.

- **Workout & Physical Activity Guidance**  
  Depending on the condition, MedAI recommends suitable exercises — whether it's complete rest, light stretches, or moderate physical activity.

- **Medication Awareness**  
  Displays a few common medications or general treatments associated with the predicted disease. This is meant purely for awareness and not as a prescription.

---

### 🧠 How It Works

1. **User selects symptoms** from a list provided in the app.
2. The system uses a **trained machine learning model** to predict the most likely disease based on the combination of symptoms.
3. MedAI then provides:
   - ⚠️ Precautions to follow  
   - 🥗 Diet suggestions  
   - 🏃‍♀️ Workout tips  
   - 💊 General medications  
4. All this information is displayed in a simple, friendly, and interactive interface.

---

### ⚠️ Disclaimer

> MedAI is developed for educational and awareness purposes only. It does **not provide medical advice**. Always consult a certified doctor for any health concerns, diagnosis, or treatment.

---

Built by **Sravani Reddy**
""")
