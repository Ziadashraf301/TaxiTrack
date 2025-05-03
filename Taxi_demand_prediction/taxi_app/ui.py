import streamlit as st

def render_user_inputs():
    st.subheader("📋 Trip Details")
    with st.expander("📅 Date & Time", expanded=True):
        trip_month = st.slider('Month', 1, 12, value=1)
        trip_day = st.slider('Day', 1, 31, value=1)
        trip_weekday = st.radio('Weekday (0=Sunday, 6=Saturday)', list(range(7)), horizontal=True)
        trip_hour = st.slider('Hour', 0, 23, value=12)

    st.markdown("---")

    with st.expander("🚕 Trip Metadata", expanded=True):
        day_type = st.selectbox('Day Type', ['Weekday', 'Weekend'])
        vendor_label = st.selectbox('Vendor Label', ['Creative Mobile Technologies', 'VeriFone Inc.', 'Other'])
        payment_type_label = st.selectbox('Payment Type', ['Credit Card', 'Cash', 'No Charge', 'Dispute', 'Unknown', 'Voided Trip'])
        trip_type = st.radio('Trip Type', ['Short', 'Medium', 'Long'])

    st.markdown("---")

    with st.expander("📊 Trip Averages", expanded=True):
        avg_distance = st.number_input('Average Distance (in miles)', min_value=0.0, format="%.2f")
        avg_passenger_count = st.number_input('Average Passenger Count', min_value=0, max_value=10, step=1)
        unique_pickup_locations = st.number_input('Unique Pickup Locations', min_value=1)

    return {
        'trip_month': trip_month,
        'trip_day': trip_day,
        'trip_weekday': trip_weekday,
        'trip_hour': trip_hour,
        'day_type': day_type,
        'vendor_label': vendor_label,
        'payment_type_label': payment_type_label,
        'trip_type': trip_type,
        'avg_distance': avg_distance,
        'avg_passenger_count': avg_passenger_count,
        'unique_pickup_locations': unique_pickup_locations
    }
