import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model, encoder, and scaler
model = pickle.load(open("model.pkl", "rb"))
encoder = pickle.load(open("encoder.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("Car Price Prediction App")

st.write("""
Provide the car features below to predict the estimated price of the car.
""")

# User Inputs
make = st.selectbox("Make", [
    'alfa-romero', 'audi', 'bmw', 'chevrolet', 'dodge', 'honda', 'isuzu', 'jaguar',
    'mazda', 'mercedes-benz', 'mercury', 'mitsubishi', 'nissan', 'peugot', 'plymouth',
    'porsche', 'renault', 'saab', 'subaru', 'toyota', 'volkswagen', 'volvo'
])
fuel_type = st.selectbox("Fuel Type", ['gas', 'diesel'])
body_style = st.selectbox("Body Style", ['convertible', 'hatchback', 'sedan', 'wagon', 'hardtop'])
drive_wheels = st.selectbox("Drive Wheels", ['rwd', 'fwd', '4wd'])
engine_location = st.selectbox("Engine Location", ['front', 'rear'])
engine_type = st.selectbox("Engine Type", ['dohc', 'ohcv', 'ohc', 'l', 'rotor', 'ohcf', 'dohcv'])

symboling = st.number_input("Symboling", min_value=-2, max_value=3, value=0)
normalized_losses = st.number_input("Normalized Losses", min_value=65.0, max_value=256.0, value=115.0)
width = st.number_input("Width (inches)", min_value=60.0, max_value=75.0, value=66.0)
height = st.number_input("Height (inches)", min_value=48.0, max_value=60.0, value=54.0)
engine_size = st.number_input("Engine Size", min_value=61, max_value=326, value=120)
horsepower = st.number_input("Horsepower", min_value=48, max_value=288, value=95)
city_mpg = st.number_input("City MPG", min_value=13, max_value=49, value=24)
highway_mpg = st.number_input("Highway MPG", min_value=16, max_value=54, value=30)

if st.button("Predict Price"):
    input_df = pd.DataFrame({
        'symboling': [symboling],
        'normalized-losses': [np.log(normalized_losses)],
        'make': [make],
        'fuel-type': [fuel_type],
        'body-style': [body_style],
        'drive-wheels': [drive_wheels],
        'engine-location': [engine_location],
        'width': [width],
        'height': [height],
        'engine-type': [engine_type],
        'engine-size': [engine_size],
        'horsepower': [horsepower],
        'city-mpg': [city_mpg],
        'highway-mpg': [highway_mpg],
    })

    # Encode categorical columns
    cat_cols = ['make', 'fuel-type', 'body-style', 'drive-wheels', 'engine-location', 'engine-type']
    input_df[cat_cols] = encoder.transform(input_df[cat_cols])

    # Apply scaling to all features
    input_scaled = scaler.transform(input_df)

    # Predict price
    price_pred = model.predict(input_scaled)[0]
    st.success(f"Predicted Car Price: ${price_pred:,.2f}")
