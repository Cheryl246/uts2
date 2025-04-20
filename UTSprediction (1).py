import streamlit as st
import pickle
import pandas as pd
import numpy as np
import joblib

model = joblib.load('Ranfor_train1 (1).pkl')
target_encoded = joblib.load('target_encoded.pkl')
meal_plan = joblib.load('meal_plan.pkl')
room_type = joblib.load('room_type.pkl')
market_segment = joblib.load('market_segment (2).pkl')

def preprocess_data(data):
    df = pd.DataFrame([data], columns=['Booking ID', 'Number of adults', 'Number of children', 
                                       'Number of weekend nights', 'Number of week nights', 'Type of meal plan', 
                                       'Required car parking space', 'Room type reserved', 'Lead time', 
                                       'Arrival year', 'Arrival month', 'Arrival date', 'Market segment type', 
                                       'Repeated guest', 'Number of previous cancellations', 
                                       'Number of previous bookings not canceled', 'Average price per room', 
                                       'Number of special requests'])

    df = df.replace(meal_plan)
    df = df.replace(room_type)
    df = pd.get_dummies(df, columns=['Market segment type'], drop_first=True)
    
    df.drop(['Booking ID'], axis=1, inplace=True)
    df.replace('', np.nan, inplace=True)
    df.fillna(df.median(), inplace=True)

    df = df.apply(pd.to_numeric, errors='coerce')
    df.fillna(df.median(), inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.median(), inplace=True)
    
    return df

def make_prediction(features):
    input_array = np.array(features).reshape(1, -1)
    if np.any(np.isnan(input_array)) or np.any(np.isinf(input_array)):
        raise ValueError("Input contains NaN or infinite values.")
    return model.predict(input_array)[0]

def main():
    st.title('Hotel Reservation Prediction')

    data = {
        'Booking ID': st.text_input("Booking_ID"),
        'Number of adults': st.selectbox("Adults", options=range(1, 11)),
        'Number of children': st.selectbox("Children, under 17", options=range(1, 11)),
        'Number of weekend nights': st.number_input("Weekend Nights", 0, 8),
        'Number of week nights': st.number_input("Week Nights (Mon to Fri)", 0, 7),
        'Type of meal plan': st.selectbox("Type of meal plan", ["Meal Plan 1", "Meal Plan 2", "Meal Plan 3", "Not Selected"]),
        'Required car parking space': st.radio("Required car parking space", [0, 1]),
        'Room type reserved': st.selectbox("Room type reserved", ["Room_Type 1", "Room_Type 2", "Room_Type 3", "Room_Type 4", "Room_Type 5", "Room_Type 6", "Room_Type 7"]),
        'Lead time': st.number_input("Lead time (in days)", 0, 365),
        'Arrival year': st.number_input("Arrival year", 2016, 2020),
        'Arrival month': st.number_input("Arrival month", 1, 12),
        'Arrival date': st.number_input("Arrival date", 1, 31),
        'Market segment type': st.radio("Market segment type", ["Online", "Corporate", "Complementary", "Offline", "Aviation"]),
        'Repeated guest': st.radio("Repeated guest", [0, 1]),
        'Number of previous cancellations': st.number_input("Number of previous cancellations", 0, 30),
        'Number of previous bookings not canceled': st.number_input("Number of previous bookings not canceled", 0, 30),
        'Average price per room': st.number_input("Average price per room (in Euros)", 0.0, 1000.0),
        'Number of special requests': st.number_input("Number of special requests", 0, 10)
    }

    df = preprocess_data(data)

    if st.button('Make Prediction'):
        result = make_prediction(df)
        st.success(f'The prediction is: {result}')

if __name__ == '__main__':
    main()
