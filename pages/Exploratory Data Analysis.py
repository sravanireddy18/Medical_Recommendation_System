import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Set page config
st.set_page_config(page_title="EDA Visualizations", layout="wide")

# Load datasets
training = pd.read_csv(r"C:\Users\bsrav\INNOMATICS\INNOMATICS_NOTEBOOK\PROJECT\prj\archive\training.csv")
description_df = pd.read_csv(r"C:\Users\bsrav\INNOMATICS\INNOMATICS_NOTEBOOK\PROJECT\prj\archive\description.csv")
diets_df = pd.read_csv(r"C:\Users\bsrav\INNOMATICS\INNOMATICS_NOTEBOOK\PROJECT\prj\archive\diets.csv")
medications_df = pd.read_csv(r"C:\Users\bsrav\INNOMATICS\INNOMATICS_NOTEBOOK\PROJECT\prj\archive\medications.csv")
precautions_df = pd.read_csv(r"C:\Users\bsrav\INNOMATICS\INNOMATICS_NOTEBOOK\PROJECT\prj\archive\precautions_df.csv")
workout_df = pd.read_csv(r"C:\Users\bsrav\INNOMATICS\INNOMATICS_NOTEBOOK\PROJECT\prj\archive\workout_df.csv")
# Load symptom severity data
symptom_severity = pd.read_csv(r"C:\Users\bsrav\INNOMATICS\INNOMATICS_NOTEBOOK\PROJECT\prj\archive\Symptom-severity.csv")

# Title
st.title("📊 Exploratory Data Analysis (EDA)")
st.write("This page contains visualizations and insights for disease patterns, symptom severity, and more.")

# Sidebar options
st.sidebar.title("📌 Select Visualization")
selected_chart = st.sidebar.radio("Choose a visualization:", [
    "Disease Frequency",
    "Symptom Severity Distribution",
    "Common Symptoms per Disease",
    "Precautions Word Cloud",
    "Distribution of Symptoms per Disease",
    "Correlation Heatmap of Symptoms",
    "Diet Recommendations Word Cloud"
])

# 1️⃣ Disease Frequency
if selected_chart == "Disease Frequency":
    st.subheader("📌 Disease Frequency in Training Data")
    fig, ax = plt.subplots(figsize=(12, 6))
    training['prognosis'].value_counts().plot(kind='bar', color='skyblue', ax=ax)
    plt.xlabel("Disease")
    plt.ylabel("Frequency")
    plt.xticks(rotation=90)
    st.pyplot(fig)

    # 🔍 Insights
    st.markdown("""
    **Insights:**
    - This bar chart shows that the training dataset has an equal number (120) of samples for each disease. 
    - This indicates a perfectly balanced dataset, which is helpful for training machine learning models.
    """)

# 2️⃣ Symptom Severity Distribution
elif selected_chart == "Symptom Severity Distribution":
    st.subheader("📌 Symptom Severity Distribution")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(symptom_severity['weight'], bins=10, kde=True, color='salmon', ax=ax)
    plt.xlabel("Symptom Severity")
    plt.ylabel("Count")
    st.pyplot(fig)

    # 🔍 Insights
    st.markdown("""
    **Insights:**
    - This chart shows that most symptoms fall in the moderate severity range (4-5), with fewer symptoms at very low or very high severity levels.
    """)

# 3️⃣ Common Symptoms per Disease (Heatmap)
elif selected_chart == "Common Symptoms per Disease":
    st.subheader("📌 Common Symptoms per Disease")
    symptom_columns = training.columns[:-1]  # Exclude disease column
    symptom_counts = training[symptom_columns].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(symptom_counts.to_frame().T, cmap='Reds', annot=False, ax=ax)
    plt.xlabel("Symptoms")
    plt.xticks(rotation=90)
    st.pyplot(fig)

    # 🔍 Insights
    st.markdown("""
    **Insights:**
    - This heatmap shows the frequency of symptoms across different diseases. Darker red indicates a symptom is more common.  
    - Fatigue, chest pain, abdominal pain, loss of appetite, itching, and breathlessness are the most frequently reported symptoms. Less common symptoms are shown in lighter shades.
    """)

# 4️⃣ Precautions Word Cloud
elif selected_chart == "Precautions Word Cloud":
    st.subheader("📌 Precautions Word Cloud")
    precaution_text = " ".join(precautions_df.iloc[:, 1:].fillna('').values.flatten())
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(precaution_text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)

    # 🔍 Insights
    st.markdown("""
    **Insights:**
    - This word cloud about precautions emphasizes consulting a doctor, avoiding unhealthy things (like fatty foods), eating healthy, maintaining hygiene, and taking medication.
    """)

# 5️⃣ Distribution of Symptoms per Disease
elif selected_chart == "Distribution of Symptoms per Disease":
    st.subheader("📌 Distribution of Symptoms per Disease")
    disease_counts = training.iloc[:, :-1].sum(axis=1).value_counts()
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=disease_counts.index, y=disease_counts.values, palette='coolwarm', ax=ax)
    plt.xlabel("Number of Symptoms")
    plt.ylabel("Number of Diseases")
    st.pyplot(fig)

    # 🔍 Insights
    st.markdown("""
    **Insights:**
    - This chart shows that most diseases have between 4 and 6 symptoms.  
    - There are fewer diseases with very few or very many symptoms.
    """)

# 6️⃣ Correlation Heatmap of Symptoms
elif selected_chart == "Correlation Heatmap of Symptoms":
    st.subheader("📌 Correlation Heatmap of Symptoms")
    fig, ax = plt.subplots(figsize=(14, 10))
    symptom_corr = training.iloc[:, :-1].corr()
    sns.heatmap(symptom_corr, cmap='coolwarm', linewidths=0.5, ax=ax)
    st.pyplot(fig)

    # 🔍 Insights
    st.markdown("""
    **Insights:**
    - This is a symptom correlation heatmap showing how often symptoms appear together. 
    - Red indicates strong correlation, blue indicates negative correlation, and white indicates no correlation. 
    - It helps identify potential underlying conditions based on symptom patterns.
    """)

# 7️⃣ Diet Recommendations Word Cloud
elif selected_chart == "Diet Recommendations Word Cloud":
    st.subheader("📌 Diet Recommendations Word Cloud")
    diet_text = " ".join(diets_df.iloc[:, 1:].stack().dropna().astype(str))
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(diet_text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)

    # 🔍 Insights
    st.markdown("""
    **Insights:**
    - This word cloud highlights that diet recommendations often focus on adjusting intake (low/high), emphasizing whole foods (fruits, vegetables, grains, protein), and addressing specific health conditions.
    """)

