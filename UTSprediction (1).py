#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pickle
import pandas as pd
import numpy as np
import joblib


# In[2]:


# Load the machine learning model and encode
model = joblib.load('Ranfor_train1.pkl')
target_encoded= joblib.load('target_encoded.pkl')
meal_plane=joblib.load('meal_plan.pkl')
room_type=joblib.load('room_type.pkl')
market_segment=joblib.load('market_segment.pkl')


# In[4]:


def main():
    st.title('Hotel Reservastion Prediction')
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
    arrival_month = st.selectbox("Arrival month", ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"])
    arrival_date = st.date_input("Arrival date")
    market_segment_type = st.radio("Market segment type", ["Online", "Corporate", "Complementary", "Offline","Aviation"])
    repeated_guest = st.radio("Repeated guest", [0, 1])
    no_of_previous_cancellations = st.number_input("Number of previous cancellations", 0, 30)
    no_of_previous_bookings_not_canceled = st.number_input("Number of previous bookings not canceled", 0, 30)
    avg_price_per_room = st.text_input("Average price per room (in Euros)")
    no_of_special_requests = st.number_input("Number of special requests", 0, 10)
    
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
        
    df = pd.DataFrame([list(data.values())], columns=['Booking ID', 'Number of adults', 'Number of children', 'Number of weekend nights', 
                                                  'Number of week nights', 'Type of meal plan', 'Required car parking space', 
                                                  'Room type reserved', 'Lead time', 'Arrival year', 'Arrival month', 'Arrival date', 
                                                  'Market segment type', 'Repeated guest', 'Number of previous cancellations', 
                                                  'Number of previous bookings not canceled', 'Average price per room', 
                                                  'Number of special requests'])
    df = df.replace(meal_plan)
    df = df.replace(room_type)
    cat_market = df[['Market segment type']]
    cat_enc_market=pd.DataFrame(oneHot_encode_market.transform(cat_market).toarray(),columns=oneHot_encode_market.get_feature_names_out())
    df=pd.concat([df,cat_enc_market], axis=1)
    df=df.drop(['Market segment type'],axis=1)

    if st.button('Make Prediction'):
        features=df      
        result = make_prediction(features)
        st.success(f'The prediction is: {result}')

def make_prediction(features):
    input_array = np.array(features).reshape(1, -1)
    prediction = model.predict(input_array)
    return prediction[0]

if __name__ == '__main__':
    main()


# In[ ]:




