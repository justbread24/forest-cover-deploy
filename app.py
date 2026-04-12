import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load your trained model (cached for speed)
@st.cache_resource
def load_model():
    return joblib.load("smote_rf_streamlit.joblib")  # New filename

st.set_page_config(page_title="Forest Cover Predictor", page_icon="🌲", layout="wide")

st.title("🌲 Forest Cover Type Predictor")
st.markdown("Enter terrain features to predict forest cover type (1-7) using SMOTE + Random Forest")

# Load model
model = load_model()

# Define expected feature columns (exactly matching your training data)
FEATURE_COLS = [
    'Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 
    'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways', 
    'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 
    'Horizontal_Distance_To_Fire_Points', 'Wilderness_Area1', 'Wilderness_Area2', 
    'Wilderness_Area3', 'Wilderness_Area4', 'Soil_Type1', 'Soil_Type2', 
    'Soil_Type3', 'Soil_Type4', 'Soil_Type5', 'Soil_Type6', 'Soil_Type7', 
    'Soil_Type8', 'Soil_Type9', 'Soil_Type10', 'Soil_Type11', 'Soil_Type12', 
    'Soil_Type13', 'Soil_Type14', 'Soil_Type15', 'Soil_Type16', 'Soil_Type17', 
    'Soil_Type18', 'Soil_Type19', 'Soil_Type20', 'Soil_Type21', 'Soil_Type22', 
    'Soil_Type23', 'Soil_Type24', 'Soil_Type25', 'Soil_Type26', 'Soil_Type27', 
    'Soil_Type28', 'Soil_Type29', 'Soil_Type30', 'Soil_Type31', 'Soil_Type32', 
    'Soil_Type33', 'Soil_Type34', 'Soil_Type35', 'Soil_Type36', 'Soil_Type37', 
    'Soil_Type38', 'Soil_Type39', 'Soil_Type40'
]

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("🌄 Terrain Features")
    elevation = st.number_input("Elevation (meters)", 1800.0, 3900.0, 3000.0, 1)
    aspect = st.number_input("Aspect (0-360°)", 0, 360, 150, 1)
    slope = st.number_input("Slope (degrees)", 0.0, 66.0, 15.0, 0.1)
    horiz_hyd = st.number_input("Horizontal Distance to Hydrology (meters)", 0.0, 1400.0, 200.0, 1)
    vert_hyd = st.number_input("Vertical Distance to Hydrology (meters)", -170.0, 400.0, 0.0, 1)
    horiz_road = st.number_input("Horizontal Distance to Roadways (meters)", 0.0, 7100.0, 500.0, 1)
    
with col2:
    st.subheader("☀️ Hillshade & Fire")
    hillshade_9am = st.number_input("Hillshade 9am (0-255)", 0, 254, 200, 1)
    hillshade_noon = st.number_input("Hillshade Noon (0-255)", 0, 254, 210, 1)
    hillshade_3pm = st.number_input("Hillshade 3pm (0-255)", 0, 254, 190, 1)
    fire_dist = st.number_input("Horizontal Distance to Fire Points (meters)", 0.0, 7200.0, 800.0, 1)

# Wilderness Areas (binary 0/1)
st.subheader("🏞️ Wilderness Area (select one)")
wilderness = st.radio("Wilderness Area", ["None", "Area 1", "Area 2", "Area 3", "Area 4"], index=0)
wilderness_map = {"None": 0, "Area 1": 1, "Area 2": 0, "Area 3": 0, "Area 4": 0}
wa_vals = [wilderness_map[wilderness]] + [0, 0, 0]

# Soil Types (binary 0/1 - let user pick one for simplicity)
st.subheader("⛰️ Soil Type (select one)")
soil_types = [f"Soil_Type{i+1}" for i in range(40)]
soil_idx = st.selectbox("Soil Type", soil_types, index=0)
soil_vals = [0] * 40
soil_vals[soil_types.index(soil_idx)] = 1

# Combine all features
features = [elevation, aspect, slope, horiz_hyd, vert_hyd, horiz_road, 
           hillshade_9am, hillshade_noon, hillshade_3pm, fire_dist] + wa_vals + soil_vals

# Create input DataFrame with exact column names
input_df = pd.DataFrame([features], columns=FEATURE_COLS)

# Prediction button
if st.button("🌲 Predict Forest Cover Type", type="primary"):
    with st.spinner("Predicting..."):
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        
    st.success(f"**Predicted Cover Type: {int(prediction)}**")
    
    # Show probabilities
    st.subheader("Prediction Confidence")
    prob_df = pd.DataFrame({
        "Cover Type": range(1, 8),
        "Probability": probabilities
    }).sort_values("Probability", ascending=False)
    
    st.bar_chart(prob_df.set_index("Cover Type"))
    
    st.caption("Cover Types: 1-Spruce/Fir, 2-Lodgepole Pine, 3-Ponderosa Pine, 4-Cottonwood/Willow, 5-Aspen, 6-Douglas-fir, 7-Krummholz")

# Sidebar info
with st.sidebar:
    st.info("**Model**: SMOTE + Random Forest (50 trees)")
    st.info("**Features**: 54 terrain/wilderness/soil variables")
    st.info("**Training**: Stratified splits + cross-validation")
    st.info("**[Source Code](https://github.com/YOUR_USERNAME/forest-cover-predictor)")
