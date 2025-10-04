import pandas as pd
import numpy as np
import os

df = pd.read_csv("insurance.csv")

tier_1_cities = {"Mumbai", "Delhi", "Bangalore", "Chennai", "Kolkata", "Hyderabad", "Pune"}
tier_2_cities = {
    "Jaipur", "Chandigarh", "Indore", "Lucknow", "Patna", "Ranchi", "Visakhapatnam", "Coimbatore",
    "Bhopal", "Nagpur", "Vadodara", "Surat", "Rajkot", "Jodhpur", "Raipur", "Amritsar", "Varanasi",
    "Agra", "Dehradun", "Mysore", "Jabalpur", "Guwahati", "Thiruvananthapuram", "Ludhiana", "Nashik",
    "Allahabad", "Udaipur", "Aurangabad", "Hubli", "Belgaum", "Salem", "Vijayawada", "Tiruchirappalli",
    "Bhavnagar", "Gwalior", "Dhanbad", "Bareilly", "Aligarh", "Gaya", "Kozhikode", "Warangal",
    "Kolhapur", "Bilaspur", "Jalandhar", "Noida", "Guntur", "Asansol", "Siliguri"
}


def bmi(weight, height) -> float:
    return weight/(height**2)
    
def lifestyle_risk(smoker, bmi) -> str:
    if smoker and bmi > 30:
        return "high"
    elif smoker or bmi > 27:
        return "medium"
    else:
        return "low"

def age_group(age) -> str:
    if age < 25:
        return "young"
    elif age < 45:
        return "adult"
    elif age < 60:
        return "middle_aged"
    return "senior"


def city_tier(city) -> int:
    city = city.capitalize()
    if city in tier_1_cities:
        return "1"
    elif city in tier_2_cities:
        return "2"
    else:
        return "3"
    
df["bmi"] = df[["weight", "height"]].apply(lambda x: bmi(x[0], x[1]), axis=1)
df["lifestyle_risk"] = df[["smoker", "bmi"]].apply(lambda x: lifestyle_risk(x[0], x[1]), axis = 1)
df["age_group"] = df["age"].apply(lambda x: age_group(x))
df["city_tier"] = df["city"].apply(lambda x: city_tier(x))

df = df.drop(["weight", "height", "age", "smoker", "city"], axis = 1)
print(df.columns)

X = df.drop("insurance_premium_category", axis=1)
y = df["insurance_premium_category"]

os.makedirs("data/raw", exist_ok=True)

X.to_csv("data/raw/X.csv")
y.to_csv("data/raw/y.csv")
