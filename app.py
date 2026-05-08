import streamlit as st

from src.predict import predict


st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 House Price Prediction App")

st.write(
    "This app predicts house prices based on property features such as area, rooms, parking, location-related features, and furnishing status."
)

st.subheader("House Information")

area = st.number_input(
    "Area",
    min_value=500,
    max_value=20000,
    value=5000
)

bedrooms = st.number_input(
    "Bedrooms",
    min_value=1,
    max_value=10,
    value=3
)

bathrooms = st.number_input(
    "Bathrooms",
    min_value=1,
    max_value=10,
    value=2
)

stories = st.number_input(
    "Stories",
    min_value=1,
    max_value=5,
    value=2
)

parking = st.number_input(
    "Parking Spaces",
    min_value=0,
    max_value=5,
    value=1
)

mainroad = st.selectbox(
    "Main Road Access",
    options=["yes", "no"]
)

guestroom = st.selectbox(
    "Guestroom",
    options=["yes", "no"]
)

basement = st.selectbox(
    "Basement",
    options=["yes", "no"]
)

hotwaterheating = st.selectbox(
    "Hot Water Heating",
    options=["yes", "no"]
)

airconditioning = st.selectbox(
    "Air Conditioning",
    options=["yes", "no"]
)

prefarea = st.selectbox(
    "Preferred Area",
    options=["yes", "no"]
)

furnishingstatus = st.selectbox(
    "Furnishing Status",
    options=["furnished", "semi-furnished", "unfurnished"]
)

if st.button("Predict House Price"):
    input_data = {
        "area": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "stories": stories,
        "parking": parking,
        "mainroad": mainroad,
        "guestroom": guestroom,
        "basement": basement,
        "hotwaterheating": hotwaterheating,
        "airconditioning": airconditioning,
        "prefarea": prefarea,
        "furnishingstatus": furnishingstatus
    }

    predicted_price = predict(input_data)

    st.subheader("Prediction Result")

    st.metric(
        label="Predicted House Price",
        value=f"{predicted_price:,.0f}"
    )

    st.info(
        "This prediction is an estimate based on the trained regression model and the selected property features."
    )