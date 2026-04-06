# =============================================
# ECOTYPE - FOREST COVER TYPE PREDICTION APP
# Streamlit Web Application
# =============================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ── Page Configuration ──────────────────────
st.set_page_config(
    page_title="EcoType - Forest Cover Predictor",
    page_icon="🌲",
    layout="wide"
)

# ── Load Models and Encoders ─────────────────
@st.cache_resource
def load_models():
    base_path = os.path.dirname(os.path.abspath(__file__))
    models_path = os.path.join(base_path, '..', 'models')

    model        = joblib.load(os.path.join(models_path, 'best_model.pkl'))
    le_wilderness= joblib.load(os.path.join(models_path, 'le_wilderness.pkl'))
    le_soil      = joblib.load(os.path.join(models_path, 'le_soil.pkl'))
    le_target    = joblib.load(os.path.join(models_path, 'le_target.pkl'))

    return model, le_wilderness, le_soil, le_target

model, le_wilderness, le_soil, le_target = load_models()

# ── Header ───────────────────────────────────
st.title("🌲 EcoType — Forest Cover Type Predictor")
st.markdown("### Predict the type of forest cover using cartographic data")
st.markdown("---")

# ── Sidebar Info ─────────────────────────────
with st.sidebar:
    st.markdown("""
<div style='background: linear-gradient(135deg, #1a472a, #2d6a4f);
            padding: 20px; 
            border-radius: 10px; 
            text-align: center;'>
    <h1 style='color: white; font-size: 50px;'>🌲🌿🌲</h1>
    <h3 style='color: #b7e4c7;'>Forest Cover</h3>
    <h3 style='color: #b7e4c7;'>Type Predictor</h3>
</div>
""", unsafe_allow_html=True)
    st.markdown("## 🌿 About EcoType")
    st.markdown("""
    This app predicts **forest cover type** based on 
    cartographic variables like elevation, slope, 
    soil type, and wilderness area.
    
    **Model:** Random Forest Classifier  
    **Accuracy:** 99.63%  
    **Classes:** 7 Forest Types  
    """)
    st.markdown("---")
    st.markdown("**7 Forest Cover Types:**")
    forest_types = [
        "🌲 Spruce/Fir",
        "🌲 Lodgepole Pine",
        "🌲 Ponderosa Pine",
        "🌿 Cottonwood/Willow",
        "🍃 Aspen",
        "🌲 Douglas-fir",
        "🏔️ Krummholz"
    ]
    for ft in forest_types:
        st.markdown(f"- {ft}")

# ── Input Section ────────────────────────────
st.markdown("## 📋 Enter Land Information")
st.markdown("Fill in the values below and click **Predict** to get the forest type.")

# Two columns layout
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🏔️ Terrain Features")

    elevation = st.number_input(
        "Elevation (meters)",
        min_value=0, max_value=4000,
        value=2500,
        help="Height above sea level in meters"
    )

    aspect = st.number_input(
        "Aspect (degrees 0-360)",
        min_value=0, max_value=360,
        value=150,
        help="Direction the slope faces (0=North, 90=East, 180=South, 270=West)"
    )

    slope = st.number_input(
        "Slope (degrees)",
        min_value=0, max_value=90,
        value=15,
        help="Steepness of terrain in degrees"
    )

    h_dist_hydrology = st.number_input(
        "Horizontal Distance to Hydrology (m)",
        min_value=0, max_value=2000,
        value=300,
        help="Horizontal distance to nearest water source"
    )

    v_dist_hydrology = st.number_input(
        "Vertical Distance to Hydrology (m)",
        min_value=-500, max_value=500,
        value=50,
        help="Vertical distance to nearest water source (negative = below water level)"
    )

    h_dist_roadways = st.number_input(
        "Horizontal Distance to Roadways (m)",
        min_value=0, max_value=8000,
        value=1500,
        help="Horizontal distance to nearest road"
    )

with col2:
    st.markdown("### ☀️ Sunlight & Environment")

    hillshade_9am = st.slider(
        "Hillshade at 9am (0-255)",
        min_value=0, max_value=255,
        value=220,
        help="Sunlight index at 9AM (0=dark, 255=bright)"
    )

    hillshade_noon = st.slider(
        "Hillshade at Noon (0-255)",
        min_value=0, max_value=255,
        value=200,
        help="Sunlight index at Noon"
    )

    hillshade_3pm = st.slider(
        "Hillshade at 3pm (0-255)",
        min_value=0, max_value=255,
        value=140,
        help="Sunlight index at 3PM"
    )

    h_dist_fire = st.number_input(
        "Horizontal Distance to Fire Points (m)",
        min_value=0, max_value=8000,
        value=1500,
        help="Distance to nearest wildfire ignition point"
    )

    wilderness_area = st.selectbox(
        "Wilderness Area",
        options=le_wilderness.classes_,
        help="The designated wilderness area"
    )

    soil_type = st.selectbox(
        "Soil Type",
        options=le_soil.classes_,
        help="The soil classification at this location"
    )

# ── Derived Features (same as Notebook 4) ───
mean_dist_hydrology = (h_dist_hydrology + v_dist_hydrology) / 2
mean_hillshade      = (hillshade_9am + hillshade_noon + hillshade_3pm) / 3
elev_road_ratio     = elevation / (h_dist_roadways + 1)
hillshade_diff      = hillshade_9am - hillshade_3pm

# ── Encode Categorical Inputs ────────────────
wilderness_encoded = le_wilderness.transform([wilderness_area])[0]
soil_encoded       = le_soil.transform([soil_type])[0]

# ── Build Input DataFrame ────────────────────
input_data = pd.DataFrame({
    'Elevation'                          : [elevation],
    'Aspect'                             : [aspect],
    'Slope'                              : [slope],
    'Horizontal_Distance_To_Hydrology'   : [h_dist_hydrology],
    'Vertical_Distance_To_Hydrology'     : [v_dist_hydrology],
    'Horizontal_Distance_To_Roadways'    : [h_dist_roadways],
    'Hillshade_9am'                      : [hillshade_9am],
    'Hillshade_Noon'                     : [hillshade_noon],
    'Hillshade_3pm'                      : [hillshade_3pm],
    'Horizontal_Distance_To_Fire_Points' : [h_dist_fire],
    'Wilderness_Area'                    : [wilderness_encoded],
    'Soil_Type'                          : [soil_encoded],
    'Mean_Distance_To_Hydrology'         : [mean_dist_hydrology],
    'Mean_Hillshade'                     : [mean_hillshade],
    'Elevation_To_Roadway_Ratio'         : [elev_road_ratio],
    'Hillshade_Diff'                     : [hillshade_diff]
})

# ── Predict Button ───────────────────────────
st.markdown("---")
col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])

with col_btn2:
    predict_button = st.button(
        "🌲 PREDICT FOREST TYPE",
        use_container_width=True
    )

# ── Show Prediction ──────────────────────────
if predict_button:
    # Make prediction
    prediction_encoded = model.predict(input_data)[0]
    prediction_proba   = model.predict_proba(input_data)[0]

    # Decode prediction back to forest name
    predicted_forest = le_target.inverse_transform([prediction_encoded])[0]

    # Confidence score
    confidence = prediction_proba[prediction_encoded] * 100

    st.markdown("---")
    st.markdown("## 🎯 Prediction Result")

    # Result box
    st.success(f"### 🌲 Predicted Forest Cover Type: **{predicted_forest}**")
    st.info(f"### 🎯 Confidence: **{confidence:.2f}%**")

    # Probability chart
    st.markdown("### 📊 Probability for Each Forest Type")

    proba_df = pd.DataFrame({
        'Forest Type': le_target.classes_,
        'Probability': prediction_proba * 100
    }).sort_values('Probability', ascending=False)

    st.bar_chart(proba_df.set_index('Forest Type')['Probability'])

    # Input summary
    st.markdown("### 📋 Your Input Summary")
    summary_df = pd.DataFrame({
        'Feature': [
            'Elevation', 'Aspect', 'Slope',
            'H. Distance to Hydrology', 'V. Distance to Hydrology',
            'H. Distance to Roadways', 'Hillshade 9am',
            'Hillshade Noon', 'Hillshade 3pm',
            'H. Distance to Fire Points',
            'Wilderness Area', 'Soil Type'
        ],
        'Value': [
            elevation, aspect, slope,
            h_dist_hydrology, v_dist_hydrology,
            h_dist_roadways, hillshade_9am,
            hillshade_noon, hillshade_3pm,
            h_dist_fire, wilderness_area, soil_type
        ]
    })
    st.dataframe(summary_df, use_container_width=True)

# ── Footer ───────────────────────────────────
st.markdown("---")
st.markdown(
    "**EcoType** | Forest Cover Type Prediction | "
    "Built with ❤️ using Streamlit & Random Forest"
)