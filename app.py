"""Streamlit interface for the trained student-performance pipeline."""
import streamlit as st
from src.predict import get_training_ranges, load_model, predict

st.set_page_config(page_title="Student Performance Prediction", page_icon="🎓")
st.title("Student Performance Prediction")
st.write(
    "Estimate a Performance Index from the five dataset features. "
    "This is an educational estimate, not a high-stakes assessment."
)


@st.cache_resource
def get_model():
    return load_model()


@st.cache_data
def get_ranges():
    return get_training_ranges()


try:
    ranges = get_ranges()
except (FileNotFoundError, ValueError) as exc:
    st.error(f"Could not load the training data: {exc}")
    st.stop()

with st.form("prediction_form"):
    c1, c2 = st.columns(2)
    with c1:
        hours_min, hours_max = (int(v) for v in ranges["Hours Studied"])
        previous_min, previous_max = (int(v) for v in ranges["Previous Scores"])
        sleep_min, sleep_max = (int(v) for v in ranges["Sleep Hours"])
        hours = st.slider("Hours Studied", hours_min, hours_max, min(5, hours_max))
        previous = st.slider(
            "Previous Scores", previous_min, previous_max,
            min(max(75, previous_min), previous_max),
        )
        sleep = st.slider(
            "Sleep Hours", sleep_min, sleep_max,
            min(max(7, sleep_min), sleep_max),
        )
    with c2:
        extracurricular = st.selectbox("Extracurricular Activities", ["Yes", "No"])
        papers_min, papers_max = (
            int(v) for v in ranges["Sample Question Papers Practiced"]
        )
        papers = st.slider(
            "Sample Question Papers Practiced", papers_min, papers_max,
            min(3, papers_max),
        )
    submitted = st.form_submit_button("Predict")

if submitted:
    row = {
        "Hours Studied": hours,
        "Previous Scores": previous,
        "Extracurricular Activities": extracurricular,
        "Sleep Hours": sleep,
        "Sample Question Papers Practiced": papers,
    }
    try:
        result = float(predict(row, get_model())[0])
        st.metric("Estimated Performance Index", f"{result:.2f}")
        st.caption(
            "Predictions are model estimates and should not be interpreted "
            "as causal or definitive judgments."
        )
    except (FileNotFoundError, TypeError, ValueError, RuntimeError) as exc:
        st.error(str(exc))
