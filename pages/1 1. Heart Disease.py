 # -*- coding: utf-8 -*-


import numpy as np
import os
import pickle
import streamlit as st

#path
main_file_path=os.path.realpath(__file__)
file_path = os.path.join(os.path.dirname(main_file_path), "./pickle_models/heart_svm.pkl")

# loaded model
heart_svm_model = pickle.load(open(file_path,'rb'))




# Function for predicting

def heart_svm_prediction(input_data):
    
    #changing the input data as numpy array
    input_data_as_np_array = np.asarray(input_data)
    
    #reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_np_array.reshape(1,-1 )
    
    prediction = heart_svm_model.predict(input_data_reshaped)
    print(prediction)
    
    if(prediction[0]==1):
        return 'The person might have heart disease'
    else:
        return 'The The person does not have heart disease'

def main():
    
    # giving a title
    st.title('Heart Disease Prediction')
    st.write('Using Support Vector Machine algorithm')

    
    #input data
    Age =  st.number_input('Age')
    Sex = st.number_input('Sex')
    ChestPainType = st.number_input('Chest Pain Type')
    RestingBloodPressure = st.number_input('Resting Blood Pressure')
    SerumCholesterol =  st.number_input('Serum cholesterol level (in mg/dL)')
    FastingBloodSugar =  st.number_input('Fasting Blood Sugar')
    RestingECG =  st.number_input('Resting ECG result ( 0, 1 or 2 )')
    MaxHeartRate =  st.number_input('Max Heart Rate achieved')
    OldPeak =  st.number_input('Old Peak')
    Slope = st.number_input('Slope of peak exercise ST segment (1 = upsloping, 2 = flat 3, = downsloping)')
    CA = st.number_input(' Number of major vessels (0-3)')
    Thalassemia =  st.number_input('Thalassemia (3 = normal, 6 = fixed defect 7, = reversable defect)')
    Aniga =  st.checkbox('Exercise Induced Angina present')

    diagnosis = ''
    
    if(st.button('Heart test result')):
        diagnosis = heart_svm_prediction([Age,Sex,ChestPainType,RestingBloodPressure,SerumCholesterol,FastingBloodSugar,RestingECG,MaxHeartRate,Aniga,OldPeak,Slope,CA,Thalassemia])
        
    st.success(diagnosis)

if __name__== '__main__':
    main()




 







    
    