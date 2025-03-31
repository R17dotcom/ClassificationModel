import streamlit as st
import numpy as np
import joblib
import base64

# Function to dynamically change background image
def add_bg_from_local(image_file):
    """Adds a background image to the Streamlit app dynamically."""
    with open(image_file, "rb") as f:
        encoded_string = base64.b64encode(f.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded_string}");
        background-size: cover;
        background-position: center;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Streamlit page configuration
st.set_page_config(page_title="Employee Attrition Prediction", page_icon="📊")

# Set default session state
if "page" not in st.session_state:
    st.session_state.page = "home"

# Homepage with background
if st.session_state.page == "home":
    add_bg_from_local("home.jpg")  # Background for home page
    st.title("📊 Employee Attrition Prediction")
    st.write("Predict whether an employee is likely to leave or stay in the company.")

    if st.button("Go to Prediction"):
        st.session_state.page = "predict"
        st.rerun()

# Prediction Page
elif st.session_state.page == "predict":
    add_bg_from_local("home.jpg")  # Background for prediction page
    st.title("🔮 Predict Employee Attrition")
    st.write("Enter the details below to get a prediction.")

    # Input fields for prediction
    age = st.number_input("Age", min_value=18, max_value=65, value=30)
    years_at_company = st.number_input("Years at Company", min_value=0, max_value=40, value=5)
    years_with_manager = st.number_input("Years with Current Manager", min_value=0, max_value=20, value=3)
    total_working_years = st.number_input("Total Working Years", min_value=0, max_value=50, value=10)
    job_level = st.slider("Job Level (1-5)", 1, 5, 2)
    monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=50000, value=5000)
    work_life_balance = st.slider("Work-Life Balance (1-4)", 1, 4, 3)
    business_travel = st.radio("Business Travel", ["Non-Travel", "Travel Rarely", "Travel Frequently"])
    over_time = st.radio("Works Overtime?", ["No", "Yes"])
    monthly_rate = st.number_input("Monthly Rate", min_value=1000, max_value=50000, value=10000)
    daily_rate = st.number_input("Daily Rate", min_value=100, max_value=5000, value=500)
    hourly_rate = st.number_input("Hourly Rate", min_value=10, max_value=200, value=50)
    distance_from_home = st.number_input("Distance from Home (km)", min_value=1, max_value=50, value=10)

    # Convert categorical inputs to numerical values
    business_travel_map = {"Non-Travel": 0, "Travel Rarely": 1, "Travel Frequently": 2}
    over_time_map = {"No": 0, "Yes": 1}

    business_travel = business_travel_map[business_travel]
    over_time = over_time_map[over_time]

    # Feature Engineering
    experience_ratio = years_at_company / age
    stability_score = years_at_company / (total_working_years + 1)
    overtime_workload = (years_with_manager + years_at_company) / (total_working_years + 1)
    job_role_score = job_level * monthly_income
    travel_balance = business_travel * (4 - work_life_balance)
    seniority_ratio = years_at_company / (years_at_company + 1)

    # Predict button
    if st.button("Predict"):
        input_features = np.array([
            monthly_income, over_time, age, monthly_rate, daily_rate, 
            total_working_years, hourly_rate, distance_from_home, 
            experience_ratio, stability_score, overtime_workload, 
            job_role_score, travel_balance, seniority_ratio
        ]).reshape(1, -1)

        # Load model and predict
        model = joblib.load("employee_attrition_model.pkl")
        prediction = model.predict(input_features)

        # Show different images based on prediction
        if prediction[0] == 1:
            add_bg_from_local("leave.jpg")  # If the employee is likely to leave
            st.error("🚨 Prediction: Likely to Leave the Company")
        else:
            add_bg_from_local("yes.jpg")  # If the employee is likely to stay
            st.success("✅ Prediction: Likely to Stay in the Company")

    # Back to Home Button
    if st.button("Back to Home"):
        st.session_state.page = "home"
        st.rerun()
