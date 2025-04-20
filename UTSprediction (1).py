import pandas as pd
import numpy as np
import joblib
import streamlit as st

# Load the machine learning model and encoders
model = joblib.load('Ranfor_train1 (1).pkl')
target_encoded = joblib.load('target_encoded.pkl')
meal_plan = joblib.load('meal_plan.pkl')
room_type = joblib.load('room_type.pkl')
market_segment = joblib.load('market_segment (2).pkl')

# Preprocess data function to handle missing values
def preprocess_data(data):
    df = pd.DataFrame([list(data.values())], columns=[
        'Booking ID', 'Number of adults', 'Number of children', 
        'Number of weekend nights', 'Number of week nights', 'Type of meal plan', 
        'Required car parking space', 'Room type reserved', 'Lead time', 
        'Arrival year', 'Arrival month', 'Arrival date', 'Market segment type', 
        'Repeated guest', 'Number of previous cancellations', 
        'Number of previous bookings not canceled', 'Average price per room', 
        'Number of special requests'
    ])
    
    # Replace categorical values with encoded values
    df = df.replace(meal_plan)
    df = df.replace(room_type)
    
    # One-hot encode 'Market segment type' column manually using pd.get_dummies
    df = pd.get_dummies(df, columns=['Market segment type'], drop_first=True)

    # Handle missing values: replace empty strings with NaN, and then fill NaN with the median of each column
    df.replace('', np.nan, inplace=True)
    df.fillna(df.median(), inplace=True)  # Fill NaN values with median
    
    # Ensure all columns that need to be numeric are converted to float
    df = df.apply(pd.to_numeric, errors='coerce')  # Convert any non-numeric data to NaN and then handle it
    
    # Handle any NaN values again (if any NaNs remain after coercion, fill them with median)
    df.fillna(df.median(), inplace=True)
    
    # Handle infinite values (if any)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)  # Replace infinity with NaN
    df.fillna(df.median(), inplace=True)  # Fill NaN values again with the median

    return df

# Function to make predictions using the pre-trained model
def make_prediction(features):
    # Make sure the input is a numpy array and reshape it for prediction
    input_array = np.array(features).reshape(1, -1)

    # Check for any NaN or infinite values before passing to the model
    if np.any(np.isnan(input_array)) or np.any(np.isinf(input_array)):
        raise ValueError("Input contains NaN or infinite values.")

    # Make prediction using the trained model
    prediction = model.predict(input_array)
    return prediction[0]

# Streamlit user interface and prediction function
def main():
    st.title('Hotel Reservation Prediction')
    Booking_ID = st.text_input("Booking_ID")
    no_of_adults = st.selectbox("Adults", options=range(1,11))
    no_of_children = st.selectbox("Children, under 17", options=range(1,11))
    no_of_weekend_nights = st.number_input("Weekend Nights", 0, 8)
    no_of_week_nights = st.number_input("Week Nights (Mon to Fri)", 0, 7)
    type_of_meal_plan = st.selectbox("Type of meal plan", ["Meal Plan 1", "Meal Plan 2", "Meal Plan 3","Not Selected"])
    required_car_parking_space = st.radio("Required car parking space", [0, 1])
    room_type_reserved = st.selectbox("Room type reserved", ["Room_Type 1", "Room_Type 2", "Room_Type 3", "Room_Type 4","Room_Type 5","Room_Type 6","Room_Type 7"])
    lead_time = st.number_input("Lead time (in days)", 0, 365)
    arrival_year = st.number_input("Arrival year", 2016, 2020)
    arrival_month = st.number_input("Arrival month", 0, 13)
    arrival_date = st.number_input("Arrival date",0,31)
    market_segment_type = st.radio("Market segment type", ["Online", "Corporate", "Complementary", "Offline","Aviation"])
    repeated_guest = st.radio("Repeated guest", [0, 1])
    no_of_previous_cancellations = st.number_input("Number of previous cancellations", 0, 30)
    no_of_previous_bookings_not_canceled = st.number_input("Number of previous bookings not canceled", 0, 30)
    avg_price_per_room = st.text_input("Average price per room (in Euros)")
    no_of_special_requests = st.number_input("Number of special requests", 0, 10)

    # Collect data into a dictionary
    data = {
        'Booking ID': Booking_ID,
        'Number of adults': no_of_adults,
        'Number of children': no_of_children,
        'Number of weekend nights': no_of_weekend_nights,
        'Number of week nights': no_of_week_nights,
        'Type of meal plan': type_of_meal_plan,
        'Required car parking space': required_car_parking_space,
        'Room type reserved': room_type_reserved,
        'Lead time': lead_time,
        'Arrival year': arrival_year,
        'Arrival month': arrival_month,
        'Arrival date': arrival_date,
        'Market segment type': market_segment_type,
        'Repeated guest': repeated_guest,
        'Number of previous cancellations': no_of_previous_cancellations,
        'Number of previous bookings not canceled': no_of_previous_bookings_not_canceled,
        'Average price per room': avg_price_per_room,
        'Number of special requests': no_of_special_requests }

    # Convert the dictionary to a DataFrame
    df = pd.DataFrame([list(data.values())], columns=['Booking ID', 'Number of adults', 'Number of children', 'Number of weekend nights', 
                                                  'Number of week nights', 'Type of meal plan', 'Required car parking space', 
                                                  'Room type reserved', 'Lead time', 'Arrival year', 'Arrival month', 'Arrival date', 
                                                  'Market segment type', 'Repeated guest', 'Number of previous cancellations', 
                                                  'Number of previous bookings not canceled', 'Average price per room', 
                                                  'Number of special requests'])

    # Preprocess the data
    df = preprocess_data(data)

    # Predict the result
    if st.button('Make Prediction'):
        result = make_prediction(df)
        st.success(f'The prediction is: {result}')

# Run the Streamlit app
if __name__ == '__main__':
    main()

