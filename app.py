import streamlit as st
import pandas as pd
import pickle

st.header("Insurance Premium Category Prediction System")

with st.form(key="person"):
    age = st.number_input("Age of the person")
    weight = st.number_input("Weight of the person")
    height = st.number_input("Height of the person")
    income_lpa = st.number_input("Salary (in LPA) of the person")
    smoker_input = st.selectbox("Does the person smoke?", options=["True", "False"])
    city = st.text_input("City of the person")
    occupation = st.selectbox(
        "Choose the occupation of the person",
        options=["freelancer", "retired", "student", "private_job", "unemployed", "business_owner", "government_job"]
    )

    submit_button = st.form_submit_button("Submit")

with open("data/model/mod.pkl", "rb") as file:
    model = pickle.load(file)

tier_1_cities = {"Mumbai", "Delhi", "Bangalore", "Chennai", "Kolkata", "Hyderabad", "Pune"}
tier_2_cities = {
    "Jaipur", "Chandigarh", "Indore", "Lucknow", "Patna", "Ranchi", "Visakhapatnam", "Coimbatore",
    "Bhopal", "Nagpur", "Vadodara", "Surat", "Rajkot", "Jodhpur", "Raipur", "Amritsar", "Varanasi",
    "Agra", "Dehradun", "Mysore", "Jabalpur", "Guwahati", "Thiruvananthapuram", "Ludhiana", "Nashik",
    "Allahabad", "Udaipur", "Aurangabad", "Hubli", "Belgaum", "Salem", "Vijayawada", "Tiruchirappalli",
    "Bhavnagar", "Gwalior", "Dhanbad", "Bareilly", "Aligarh", "Gaya", "Kozhikode", "Warangal",
    "Kolhapur", "Bilaspur", "Jalandhar", "Noida", "Guntur", "Asansol", "Siliguri"
}

def bmi_calc(weight, height) -> float:
    return weight / (height ** 2)

def lifestyle_risk(smoker: bool, bmi: float) -> str:
    if smoker and bmi > 30:
        return "high"
    elif smoker or bmi > 27:
        return "medium"
    else:
        return "low"

def age_group(age: int) -> str:
    if age < 25:
        return "young"
    elif age < 45:
        return "adult"
    elif age < 60:
        return "middle_aged"
    return "senior"

def city_tier(city: str) -> int:
    city = city.strip().capitalize()
    if city in tier_1_cities:
        return 3
    elif city in tier_2_cities:
        return 2
    else:
        return 1

if submit_button:
    try:
        smoker_bool = smoker_input == "True"
        bmi_value = bmi_calc(weight, height)
        input_data = pd.DataFrame({
            "bmi": [bmi_value],
            "lifestyle_risk": [lifestyle_risk(smoker_bool, bmi_value)],
            "age_group": [age_group(age)],
            "city_tier": [city_tier(city)],
            "income_lpa": [income_lpa],
            "occupation": [occupation]
        })

        prediction = model.predict(input_data)
        st.success(f"Predicted Premium Category: {prediction[0]}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
