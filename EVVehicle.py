# EVVehicle.py
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# --- Page setup ---
st.title("🚗 Electric Vehicle Price Prediction")
st.write("Powering Tomorrow's Electric Revolution ⚡")

# --- Load dataset ---
@st.cache_data
def load_data():
    return pd.read_csv("cars-dataset-clean.csv")

data = load_data()
st.subheader("Dataset Preview")
st.dataframe(data.head())

# --- Feature selection ---
features = ['Battery_Capacity', 'Range_km', 'Top_Speed', 'Acceleration', 'Power']
target = 'Price'

# Handle missing columns gracefully
available_features = [f for f in features if f in data.columns]
if target not in data.columns:
    st.error("⚠ The dataset must contain a 'Price' column for prediction.")
else:
    X = data[available_features]
    y = data[target]

    # --- Train-test split ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Model training ---
    model = LinearRegression()
    model.fit(X_train, y_train)

    # --- Model evaluation ---
    y_pred = model.predict(X_test)
    accuracy = r2_score(y_test, y_pred)

    st.success(f"Model Trained ✅ | R² Score: {accuracy:.2f}")

    # --- User Input Interface ---
    st.subheader("🔧 Try Your Own Prediction")
    user_input = {}
    for feature in available_features:
        user_input[feature] = st.number_input(f"Enter {feature}:", min_value=0.0)

    if st.button("Predict Price"):
        input_df = pd.DataFrame([user_input])
        price_pred = model.predict(input_df)[0]
        st.success(f"💰 Estimated EV Price: ₹{price_pred:,.2f}")
