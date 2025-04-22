# app.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle
import streamlit as st

# Load the trained model and encoders
@st.cache_resource
def load_model_and_encoders(model_path='best_booking_model.pkl', encoder_path='label_encoders.pkl'):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    with open(encoder_path, 'rb') as file:
        encoders = pickle.load(file)
    return model, encoders

model, encoders = load_model_and_encoders()

# List of categorical columns that need encoding
categorical_cols = ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type']

def preprocess_input(input_data, encoders, default_value=-1):
    """Preprocesses the input data for prediction, handling unseen categorical values."""
    df = pd.DataFrame([input_data])
    for col in categorical_cols:
        if col in df.columns and col in encoders:
            encoder = encoders[col]
            current_value = df[col].iloc[0]
            if current_value not in encoder.classes_:
                print(f"Warning: Unseen category '{current_value}' in column '{col}'. Mapping to default value: {default_value}")
                # It's crucial to handle this consistently with how your model was trained.
                # If your model expects numerical input, you should map this to a numerical value.
                # A simple approach is to use the default_value.
                df[col] = default_value
            else:
                df[col] = encoder.transform([current_value])[0]
        elif col in df.columns:
            # If no encoder, assume it's already numerical or needs different handling
            pass
    return df

def predict_cancellation(model, processed_data):
    """Makes a prediction using the loaded model."""
    prediction = model.predict(processed_data)
    return "Canceled" if prediction[0] == 1 else "Not Canceled"

# Streamlit App
st.title('Hotel Booking Cancellation Prediction')
st.write('Please input the booking details to predict if it will be canceled.')

# Input fields for the features
no_of_adults = st.number_input('Number of Adults', min_value=0, max_value=10, value=2)
no_of_children = st.number_input('Number of Children', min_value=0, max_value=10, value=0)
no_of_weekend_nights = st.number_input('Number of Weekend Nights', min_value=0, value=1)
no_of_week_nights = st.number_input('Number of Week Nights', min_value=0, value=2)
type_of_meal_plan = st.selectbox('Type of Meal Plan', ['Meal Plan 1', 'Meal Plan 2', 'Meal Plan 3', 'Not Selected'])
required_car_parking_space = st.selectbox('Required Car Parking Space', [0, 1])
room_type_reserved = st.selectbox('Room Type Reserved', ['Room Type 1', 'Room Type 2', 'Room Type 3', 'Room Type 4', 'Room Type 5', 'Room Type 6', 'Room Type 7'])
lead_time = st.number_input('Lead Time (days)', min_value=0, value=30)
arrival_year = st.number_input('Arrival Year', min_value=2020, max_value=2030, value=2024)
arrival_month = st.number_input('Arrival Month', min_value=1, max_value=12, value=7)
arrival_date = st.number_input('Arrival Date', min_value=1, max_value=31, value=15)
market_segment_type = st.selectbox('Market Segment Type', ['Online', 'Offline', 'Corporate', 'Complementary', 'Aviation'])
repeated_guest = st.selectbox('Repeated Guest', [0, 1])
no_of_previous_cancellations = st.number_input('Previous Cancellations', min_value=0, value=0)
no_of_previous_bookings_not_canceled = st.number_input('Previous Non-Canceled Bookings', min_value=0, value=0)
avg_price_per_room = st.number_input('Average Price per Room (Euro)', min_value=0.0, value=100.0)
no_of_special_requests = st.number_input('Number of Special Requests', min_value=0, value=1)

# Prediction button
if st.button('Predict Cancellation'):
    input_data = {
        'no_of_adults': no_of_adults,
        'no_of_children': no_of_children,
        'no_of_weekend_nights': no_of_weekend_nights,
        'no_of_week_nights': no_of_week_nights,
        'type_of_meal_plan': type_of_meal_plan,
        'required_car_parking_space': required_car_parking_space,
        'room_type_reserved': room_type_reserved,
        'lead_time': lead_time,
        'arrival_year': arrival_year,
        'arrival_month': arrival_month,
        'arrival_date': arrival_date,
        'market_segment_type': market_segment_type,
        'repeated_guest': repeated_guest,
        'no_of_previous_cancellations': no_of_previous_cancellations,
        'no_of_previous_bookings_not_canceled': no_of_previous_bookings_not_canceled,
        'avg_price_per_room': avg_price_per_room,
        'no_of_special_requests': no_of_special_requests
    }

    # Preprocess the input data
    processed_input = preprocess_input(input_data, encoders)

    # Make prediction
    prediction = predict_cancellation(model, processed_input)

    st.subheader('Prediction Result:')
    st.write(f'The booking is likely to be: **{prediction}**')

st.subheader('Test Cases:')

st.write('**Test Case 1:**')
st.write("""
* Number of Adults: 2
* Number of Children: 0
* Number of Weekend Nights: 1
* Number of Week Nights: 2
* Type of Meal Plan: Meal Plan 1
* Required Car Parking Space: 0
* Room Type Reserved: Room Type 1
* Lead Time (days): 30
* Arrival Year: 2023
* Arrival Month: 10
* Arrival Date: 15
* Market Segment Type: Online
* Repeated Guest: 0
* Previous Cancellations: 0
* Previous Non-Canceled Bookings: 0
* Average Price per Room (Euro): 100.5
* Number of Special Requests: 1
""")

st.write('**Test Case 2:**')
st.write("""
* Number of Adults: 1
* Number of Children: 2
* Number of Weekend Nights: 0
* Number of Week Nights: 3
* Type of Meal Plan: Not Selected
* Required Car Parking Space: 1
* Room Type Reserved: Room Type 4
* Lead Time (days): 150
* Arrival Year: 2024
* Arrival Month: 6
* Arrival Date: 20
* Market Segment Type: Offline
* Repeated Guest: 1
* Previous Cancellations: 2
* Previous Non-Canceled Bookings: 5
* Average Price per Room (Euro): 75.0
* Number of Special Requests: 0
""")
