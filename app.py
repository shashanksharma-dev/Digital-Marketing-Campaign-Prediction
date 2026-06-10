import streamlit as st
import pandas as pd
import joblib
import base64
import time




# -------------------------------
# PAGE CONFIG (MUST BE FIRST)
# -------------------------------
st.set_page_config(
    page_title="ConvertX",
    layout="centered",
    page_icon="📊"
)

# -------------------------------
# BACKGROUND IMAGE
# -------------------------------
def set_background(image_file):
    with open(image_file, "rb") as file:
        encoded = base64.b64encode(file.read()).decode()

    page_bg = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}

    .block-container {{
        background-color: rgba(0, 0, 0, 0.65);
        padding: 2rem;
        border-radius: 15px;
    }}

    h1, h2, h3, label, p {{
        color: white !important;
    }}
    </style>
    """

    st.markdown(page_bg, unsafe_allow_html=True)

set_background("assets/wallpaper.jpg")

# -------------------------------
# UI STYLING
# -------------------------------
st.markdown("""
<style>
div.stButton > button {
    background-color: rgba(255,255,255,0.2);
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 16px;
}

div[data-testid="stSlider"] {
    background-color: rgba(255,255,255,0.1);
    padding: 10px;
    border-radius: 10px;
}

div[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    height: 10px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# SVG LOGO (ROBUST RENDER)
# -------------------------------
def render_svg(svg_file):
    with open(svg_file, "r") as f:
        svg = f.read()
    b64 = base64.b64encode(svg.encode("utf-8")).decode("utf-8")
    return f"""
    <div style="display:flex; align-items:center;">
        <img src="data:image/svg+xml;base64,{b64}" width="80">
    </div>
    """

def render_logo(image_path):
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    return f"""
    <div style="display:flex; align-items:center; justify-content:center;">
        <img src="data:image/jpg;base64,{encoded}" 
             width="70"
             style="border-radius:12px; display:block;">
    </div>
    """
# -------------------------------
# NAVBAR
# -------------------------------
col1, col2 = st.columns([1, 5])

# with col1:
#     st.markdown(render_svg("assets/logo.svg"), unsafe_allow_html=True)

# with col2:
#     st.markdown("""
#     <h1 style='margin-bottom:0;'>ConvertX</h1>
#     <p style='margin-top:0; font-size:16px;'>Predict. Target. Convert.</p>
#     """, unsafe_allow_html=True)

col1, col2 = st.columns([1, 6])

with col1:
    st.markdown(render_logo("assets/logo.jpg"), unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="display:flex; flex-direction:column; justify-content:center; height:100%;">
        <h1 style='margin-bottom:5px; font-size:42px; font-weight:700; letter-spacing:1px;'>
            ConvertX
        </h1>
        <p style='margin-top:0; font-size:18px; opacity:0.85;'>
            Predict. Target. Convert.
        </p>
    </div>
    """, unsafe_allow_html=True)



# -------------------------------
# LOAD MODEL
# -------------------------------
model = joblib.load("conversion_model.pkl")
model_columns = joblib.load("model_columns.pkl")

# -------------------------------
# INPUT SECTION
# -------------------------------
st.subheader("📊 Enter Customer Details")

age = st.slider("Age", 18, 65, 30)
income = st.number_input("Income", 10000, 1000000, 50000)
ad_spend = st.number_input("Ad Spend", 0, 10000, 500)
ctr = st.slider("Click Through Rate", 0.0, 1.0, 0.1)
website_visits = st.slider("Website Visits", 1, 50, 5)
time_on_site = st.slider("Time on Site", 10, 1000, 200)
previous_purchases = st.slider("Previous Purchases", 0, 20, 1)

channel = st.selectbox("Campaign Channel", ["Email", "Social Media", "PPC"])
gender = st.selectbox("Gender", ["Male", "Female"])

# -------------------------------
# PREDICTION
# -------------------------------
if st.button("🚀 Predict Conversion"):

    with st.spinner("Analyzing customer behavior..."):
        time.sleep(1)

        # Create aligned input
        input_aligned = pd.DataFrame(0, index=[0], columns=model_columns)

        # Fill numeric values
        input_aligned['Age'] = age
        input_aligned['Income'] = income
        input_aligned['AdSpend'] = ad_spend
        input_aligned['ClickThroughRate'] = ctr
        input_aligned['WebsiteVisits'] = website_visits
        input_aligned['TimeOnSite'] = time_on_site
        input_aligned['PreviousPurchases'] = previous_purchases

        # Categorical encoding
        if f'Gender_{gender}' in input_aligned.columns:
            input_aligned[f'Gender_{gender}'] = 1

        if f'CampaignChannel_{channel}' in input_aligned.columns:
            input_aligned[f'CampaignChannel_{channel}'] = 1

        # Predict
        prob = model.predict_proba(input_aligned)[0][1]

    # -------------------------------
    # OUTPUT
    # -------------------------------
    st.subheader(f"🔮 Conversion Probability: {prob:.2%}")

    progress_value = int(prob * 100)

    # st.markdown(f"### 📊 Confidence Level: {progress_value}%")
    st.progress(progress_value)

    if prob > 0.7:
        st.success("💰 High conversion likelihood")
    elif prob > 0.4:
        st.warning("⚠️ Moderate likelihood")
    else:
        st.error("❌ Low likelihood")

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("""
<hr style="border:1px solid rgba(255,255,255,0.2)">
<p style="text-align:center; color:white;">
Made by Shashank Sharma
</p>
""", unsafe_allow_html=True)