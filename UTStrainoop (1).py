#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder  
import numpy as np
import pickle
import joblib
import xgboost as xgb
import gzip


# In[2]:


# Load the machine learning model and encode
model = joblib.load('Ranfor_train.pkl')
target_encoded= joblib.load('target_encoded.pkl')
meal_plane=joblib.load('meal_plan.pkl')
room_type=joblib.load('room_type.pkl')
market_segment=joblib.load('market_segment.pkl')


# In[3]:


# preparing data
class DataHandler:   
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.input_df = None
        self.output_df = None
    def load_data(self):
        self.data = pd.read_csv(self.file_path) 
    def create_input_output(self, booking_status):
        self.output_df = self.data[booking_status]
        self.input_df = self.data.drop(booking_status, axis=1)


# In[8]:


# model preprocessing 
class ModelPreprocessing: 
    def __init__(self, input_data, output_data):
        self.input_data = input_data
        self.output_data = output_data
        self.le = LabelEncoder()
        self.x_train, self.x_test, self.y_train, self.y_test, self.y_predict = [None] * 5
        self.model = RandomForestClassifier(random_state=42)
        self.y_predict = None  
    def SplitData(self, test_size=0.2, random_state=42):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split( 
            self.input_data, self.output_data, test_size=test_size, random_state=random_state)
    def DropColumns(self, columns_to_drop):
        self.x_train = self.x_train.drop(columns=columns_to_drop, inplace=False)  
        self.x_test = self.x_test.drop(columns=columns_to_drop, inplace=False)
    def ModeMeal(self, column_name):
        return self.x_train[column_name].mode()[0]
    def ImputeMode(self,column_name):
        mode = self.ModeMeal(column_name)
        self.x_train[column_name].fillna(mode, inplace=True)
        self.x_test[column_name].fillna(mode, inplace=True)
    def ImputeMedian(self, columns):
        for column in columns:
            median_value = self.x_train[column].median()
            self.x_train[column].fillna(median_value, inplace=True)
            self.x_test[column].fillna(median_value, inplace=True)
    def DropDuplicate(self):  
        duplicates = self.x_train.duplicated(keep='first')
        self.x_train = self.x_train[~duplicates]
        self.y_train = self.y_train[~duplicates]  
    def BinaryEncoding(self,columns):
        for column in columns:
            if column == 'type_of_meal_plan':
                self.x_train[column] = self.x_train[column].replace(meal_plane)
                self.x_test[column] = self.x_test[column].replace(meal_plane)
            elif column == 'room_type_reserved':
                self.x_train[column] = self.x_train[column].replace(room_type)
                self.x_test[column] = self.x_test[column].replace(room_type)
            else:
                print(f"Column {column} has no encoder mapping")
    def OneHot(self,column_name): 
        encoder = OneHotEncoder(sparse=False, drop='first')  
        market_enc_train = self.x_train[[column_name]]
        market_enc_test = self.x_test[[column_name]]
        market_enc_train = pd.DataFrame(encoder.fit_transform(market_enc_train), columns=encoder.get_feature_names_out())
        market_enc_test = pd.DataFrame(encoder.transform(market_enc_test), columns=encoder.get_feature_names_out())
        self.x_train = self.x_train.reset_index(drop=True)
        self.x_test = self.x_test.reset_index(drop=True)
        self.x_train = self.x_train.drop([column_name], axis=1)
        self.x_test = self.x_test.drop([column_name], axis=1)
    def BinaryEncodeOutput(self):
        binary_mapping = {'Canceled': 1, 'Not_Canceled': 0}  
        self.y_train_encoded = self.y_train.map(binary_mapping)
        self.y_test_encoded = self.y_test.map(binary_mapping)
    def TrainModel(self):
        self.model.fit(self.x_train, self.y_train_encoded)
    def PredictModel(self):
        self.y_predict = self.model.predict(self.x_test)
        return self.y_predict
    def createReport(self):
        print('\nClassification Report\n')
        print(classification_report(self.y_test_encoded, self.y_predict, target_names=['0','1']))
    def SaveGzip(self, filename):
        with gzip.open(filename, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Model berhasil disimpan dengan kompresi gzip ke {filename}")


# In[9]:


file_path = 'Dataset_B_Hotel.csv'  
data_handler = DataHandler(file_path)
data_handler.load_data()
data_handler.create_input_output('booking_status')
input_df = data_handler.input_df
output_df = data_handler.output_df

model_process = ModelPreprocessing(input_df, output_df)
model_process.SplitData()
model_process.DropColumns(['Booking_ID'])
mode_replace = model_process.ModeMeal('type_of_meal_plan')
model_process.ImputeMode('type_of_meal_plan')
model_process.ImputeMedian(['required_car_parking_space'])
model_process.ImputeMedian(['avg_price_per_room'])
model_process.DropDuplicate()
model_process.BinaryEncoding(['type_of_meal_plan', 'room_type_reserved'])
model_process.OneHot('market_segment_type')
model_process.BinaryEncodeOutput()  
model_process.TrainModel()
model_process.PredictModel()
model_process.createReport()
model_process.SaveGzip('Ranfor_train1.pkl')


# In[ ]:




