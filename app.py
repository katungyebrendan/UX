import streamlit as st
import requests
import json

# Replace this with your Render FastAPI URL
API_URL = "https://eastcoast-1.onrender.com/"

st.title("🐄 East Coast Fever Prediction")
st.subheader("Using GCN + Knowledge Distillation")

# Collect input features
st.markdown("### Input Features")

feature_names = [
    "genotype", "longitude", "latitude", "tick", "cape",
    "cattle", "bio5", "cluster"
]

features = []
for name in feature_names:
    val = st.number_input(f"{name}", value=0.0)
    features.append(val)

# Dummy edge input (required for schema but not used in prediction)
edges = [[0, 0]]

# Submit button
if st.button("Predict"):
    payload = {
        "features": features,
        "edges": edges
    }

    try:
        response = requests.post(API_URL, json=payload)
        if response.status_code == 200:
            result = response.json()
            st.success(f"Predicted Class: **{result['prediction']}**")
            st.write("Class Probabilities:")
            st.json(result["class_probabilities"])
            st.write("Teacher Prediction:")
            st.json({
                "prediction": result["teacher_prediction"],
                "probabilities": result["teacher_probabilities"]
            })
        else:
            st.error(f"Error: {response.status_code}")
            st.text(response.text)
    except Exception as e:
        st.error(f"Request failed: {e}")
