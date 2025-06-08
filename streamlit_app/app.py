import streamlit as st
import pandas as pd
import requests

st.set_page_config(page_title="Job Success Predictor", page_icon="✅")

st.title("📄 Job Application Success Predictor")
st.markdown("Upload or select resume features to see if the application is likely to succeed.")

# Load preprocessed feature data
@st.cache_data
def load_features():
    return pd.read_csv("data/processed/X_features.csv")

df = load_features()

# Let user pick a sample row
index = st.selectbox("Choose a sample resume index:", df.index)
row = df.iloc[index]
st.write("Selected features:")
st.dataframe(row.to_frame())

# Prepare payload for API
features = row.tolist()
payload = {"features": features}

# Call FastAPI backend
if st.button("Predict Success"):
    try:
        response = requests.post("http://127.0.0.1:8000/predict", json=payload)
        if response.status_code == 200:
            result = response.json()
            st.success(f"Prediction: {'✅ Success' if result['prediction'] == 1 else '❌ Not Successful'}")
            st.metric("Probability of Success", f"{result['success_probability'] * 100:.2f}%")
        else:
            st.error(f"Error from API: {response.status_code}")
    except requests.exceptions.ConnectionError:
        st.error("❌ Could not connect to API. Make sure FastAPI server is running.")
