import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="Cardiac Assessment | Heart Health",
    page_icon="🫀",
    layout="wide"
)

ROOT = Path(__file__).resolve().parent

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
:root { --ink:#16313a; --muted:#62777d; --teal:#087e83; --line:#dce8e8; }
.stApp { background:linear-gradient(180deg,#f2f8f7 0%,#f8fbfb 320px,#f8fbfb 100%); color:var(--ink); }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
/* Explicit foreground colors keep text readable under Streamlit's active theme. */
.stApp, .stApp p, .stApp label, .stApp [data-testid="stMarkdownContainer"],
.stApp [data-testid="stWidgetLabel"] p, .stApp [data-testid="stCaptionContainer"] { color:var(--ink); }
h1,h2,h3 { font-family:'Manrope',sans-serif !important; color:var(--ink); }
[data-testid="stHeader"] { background:transparent; }
[data-testid="stSidebar"] { background:#eef6f5; border-right:1px solid var(--line); }
[data-testid="stSidebar"] *, [data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p, [data-testid="stSidebar"] li { color:#16313a !important; }
[data-testid="stSidebar"] [data-testid="stAlert"] { background:#e1f1ee; border:1px solid #c5e2dc; }
[data-testid="stSidebar"] [data-testid="stAlert"] * { color:#234b4d !important; }
[data-testid="stAlert"] { background:#fff8e7; border:1px solid #f0d99c; border-radius:12px; }
[data-testid="stAlert"] * { color:#59420d !important; }
[data-testid="stForm"] { background:#fff; border:1px solid var(--line); border-radius:18px; padding:1.5rem 1.7rem; box-shadow:0 12px 35px rgba(20,60,66,.05); }
[data-testid="stForm"] label, [data-testid="stForm"] label p { color:#29454b !important; }
[data-testid="stForm"] input, [data-testid="stForm"] textarea,
[data-testid="stForm"] [data-baseweb="input"] input,
[data-testid="stForm"] [data-baseweb="select"] > div,
[data-testid="stForm"] [data-baseweb="select"] input {
    color:#16313a !important;
    -webkit-text-fill-color:#16313a !important;
    background-color:#f4f8f8 !important;
}
[data-testid="stForm"] [data-baseweb="input"] > div,
[data-testid="stForm"] [data-baseweb="select"] > div { background-color:#f4f8f8 !important; }
[data-baseweb="popover"] *, [data-baseweb="menu"] *, [role="listbox"] * {
    color:#16313a !important;
    background-color:#fff !important;
}
[data-testid="stForm"] button[aria-label="Increment"],
[data-testid="stForm"] button[aria-label="Decrement"] { color:#087e83 !important; }
[data-testid="stMetric"] { background:#fff; border:1px solid var(--line); border-radius:14px; padding:1rem 1.2rem; }
.hero { background:linear-gradient(115deg,#0d5962,#087e83 62%,#38a6a1); color:white; padding:2.15rem 2.3rem; border-radius:22px; margin:0 0 1.25rem; box-shadow:0 18px 38px rgba(8,91,96,.16); }
.hero h1 { color:white !important; font-size:2.25rem; margin:0 0 .45rem; }
.hero p { color:#e1f3f1; margin:0; font-size:1.03rem; }
.eyebrow { text-transform:uppercase; letter-spacing:.14em; font-size:.72rem; font-weight:700; color:#b9e7e2; margin-bottom:.65rem; }
.section-note { color:#62777d; margin-top:-.5rem; margin-bottom:1rem; }
div.stButton > button, div[data-testid="stFormSubmitButton"] > button { background:#087e83; color:white; border:0; border-radius:10px; font-weight:700; min-height:2.9rem; }
div.stButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover { background:#06666b; color:white; }
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return joblib.load(ROOT / "heart_disease_model.pkl")

model = load_model()
anatomy_image = ROOT / "heart_anatomy.jpg"

# ---------------- SIDEBAR ----------------
if anatomy_image.is_file():
    st.sidebar.image(str(anatomy_image), use_container_width=True)
st.sidebar.title("Cardiac Assessment")
st.sidebar.markdown("---")
st.sidebar.subheader("About this tool")
st.sidebar.write(
    "An educational machine learning project demonstrating "
    "binary classification using patient health attributes."
)

st.sidebar.markdown("### Technology Stack")
st.sidebar.write(
    "- Python\n"
    "- Pandas\n"
    "- Scikit-learn\n"
    "- XGBoost\n"
    "- Streamlit"
)

st.sidebar.info(
    "Educational use only. This application is not a medical "
    "diagnostic tool."
)

# ---------------- MAIN HEADER ----------------
if anatomy_image.is_file():
    hero_text, hero_image = st.columns([1.55, 1], vertical_alignment="center")
else:
    hero_text = st.container()
with hero_text:
    st.markdown("""
    <div class="hero">
      <div class="eyebrow">Heart health · Decision support demo</div>
      <h1>Cardiac Assessment</h1>
      <p>Enter the recorded patient and test details to view the model's classification.</p>
    </div>
    """, unsafe_allow_html=True)
if anatomy_image.is_file():
    with hero_image:
        st.image(str(anatomy_image), caption="Heart anatomy", use_container_width=True)

st.markdown("#### Patient assessment")
st.markdown('<div class="section-note">Use the values available in the patient record. Fields are grouped to support a clear review.</div>', unsafe_allow_html=True)

st.warning(
    "Educational demonstration only. This unvalidated model is not a clinical decision support system and must not guide diagnosis or treatment."
)

st.divider()

# ---------------- INPUT FORM ----------------
with st.form("patient_form"):

    st.subheader("1. Patient Information")
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(
            "Age (years)", 1, 120, 45
        )

        sex = st.selectbox(
            "Sex",
            [1, 0],
            format_func=lambda x: "Male" if x == 1 else "Female"
        )

        cp = st.selectbox(
            "Chest Pain Type",
            [1, 2, 3, 4],
            format_func=lambda x: {
                1: "Typical angina",
                2: "Atypical angina",
                3: "Non-anginal pain",
                4: "Asymptomatic"
            }[x]
        )

    with col2:
        trestbps = st.number_input(
            "Resting Blood Pressure (mm Hg)",
            50, 250, 120
        )

        chol = st.number_input(
            "Cholesterol (mg/dL)", 50, 700, 200
        )

        fbs = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dL",
            [0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes"
        )

    st.divider()
    st.subheader("2. Cardiac Test Information")

    col3, col4 = st.columns(2)

    with col3:
        restecg = st.selectbox(
            "Resting ECG Result", [0, 1, 2]
        )

        thalach = st.number_input(
            "Maximum Heart Rate Achieved",
            50, 250, 150
        )

        exang = st.selectbox(
            "Exercise-Induced Angina",
            [0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes"
        )

    with col4:
        oldpeak = st.number_input(
            "ST Depression (Oldpeak)",
            min_value=0.0,
            max_value=10.0,
            value=1.0,
            step=0.1
        )

        slope = st.selectbox(
            "ST Segment Slope", [1, 2, 3]
        )

        ca = st.selectbox(
            "Number of Major Vessels", [0, 1, 2, 3]
        )

    thal = st.selectbox(
        "Thalassemia Test Result", [3, 6, 7]
    )

    submitted = st.form_submit_button(
        "🔍 Predict",
        use_container_width=True
    )

# ---------------- PREDICTION ----------------
if submitted:

    input_data = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.divider()
    st.subheader("3. Model Output")

    if prediction == 1:
        st.error(
            "The model classified this record as positive "
            "for the heart disease target."
        )
    else:
        st.success(
            "The model classified this record as negative "
            "for the heart disease target."
        )

    st.metric(
        "Estimated Model Probability",
        f"{probability:.1%}"
    )

    st.caption(
        "This probability is a model output, not a validated "
        "clinical risk estimate. The prediction cannot diagnose "
        "or rule out heart disease."
    )

st.divider()

st.caption(
    "Heart Disease Prediction System | Machine Learning Project"
)
