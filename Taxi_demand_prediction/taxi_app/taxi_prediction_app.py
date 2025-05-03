import streamlit as st
from taxi_model import TaxiPredictionModel
from database import DatabaseHandler
from ui import render_user_inputs

def main():
    st.title("Taxi Ride Prediction")
    st.write("Enter the details of the taxi ride to predict the trips count.")

    model = TaxiPredictionModel()
    db = DatabaseHandler()

    features = render_user_inputs()

    if st.button('Make Prediction'):
        df, prediction = model.make_prediction(features)

        if df is not None and prediction is not None:
            st.write(f"Prediction: {prediction}")

            # Add 95% Prediction Interval assuming standard error ±0.18
            se = 0.18
            margin = 1.96 * se
            lower_bound = round(prediction - margin, 2)
            upper_bound = round(prediction + margin, 2)
            st.write(f"95% Prediction Interval: [{lower_bound}, {upper_bound}]")

            db.insert_prediction(df)
            st.success("Data and prediction successfully stored in the database.")
        else:
            st.error("Prediction failed. Check logs for more information.")

if __name__ == "__main__":
    main()
