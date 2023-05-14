 # -*- coding: utf-8 -*-


import numpy as np
import os
import pickle
import streamlit as st

#path
main_file_path=os.path.realpath(__file__)
file_path = os.path.join(os.path.dirname(main_file_path), "./pickle_models/diabetes_svm.pkl")

# loaded model
diabetes_logistic_loaded_model = pickle.load(open(file_path,'rb'))




# Function for predicting

def diabetes_prediction_logistic(input_data):
    
    #changing the input data as numpy array
    input_data_as_np_array = np.asarray(input_data)
    
    #reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_np_array.reshape(1,-1 )
    
    prediction = diabetes_logistic_loaded_model.predict(input_data_reshaped)
    print(prediction)
    
    if(prediction[0]==1):
        return 'The person might have diabetes'
    else:
        return 'The person is unlikely to have diabetes'

def main():
    
    # giving a title
    st.title('Diabetes Disease Prediction')
    
    #input data

    Pregnancies = st.number_input('Pregnancies')
    Glucose =  st.number_input('Glucose')
    BloodPressure =  st.number_input('Blood Pressure')
    SkinThickness =  st.number_input('Skin Thickness')
    Insulin =  st.number_input('Insulin')
    BMI =  st.number_input('BMI')
    DiabetesPedigreeFunction =  st.number_input('Diabetes Pedigree Function')
    Age =  st.number_input('Age')

    diagnosis = ''
    
    if(st.button('Diabetes test result')):
        diagnosis = diabetes_prediction_logistic([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
        
    st.success(diagnosis)

if __name__== '__main__':
    main()




 







    
    