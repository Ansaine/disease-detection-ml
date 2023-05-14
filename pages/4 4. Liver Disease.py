 # -*- coding: utf-8 -*-


import numpy as np
import os
import pickle
import streamlit as st

#path
main_file_path=os.path.realpath(__file__)
file_path = os.path.join(os.path.dirname(main_file_path), "./pickle_models/liver_lr.pkl")

# loaded model
liver_lr_loaded_model = pickle.load(open(file_path,'rb'))




# Function for predicting

def liver_prediction_lr(input_data):
    
    #changing the input data as numpy array
    input_data_as_np_array = np.asarray(input_data)
    
    #reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_np_array.reshape(1,-1 )
    
    prediction = liver_lr_loaded_model.predict(input_data_reshaped)
    print(prediction)
    
    if(prediction[0]==1):
        return 'The person might have liver disease'
    else:
        return 'The The person does not have liver disease'

def main():
    
    # giving a title
    st.title('Liver Disease Prediction')
    st.write('Using Logistic Regression algorithm')
    
    #input data
    
    age =  st.number_input('Age')
    gender = st.number_input('Gender : Enter 1 for Male and 0 for Female')
    t_bilirubin = st.number_input('Total Bilirubin')
    d_bilirubin =  st.number_input('Direct Bilirubin')
    Alkaline_phosphotase =  st.number_input('Alkaline_Phosphotase')
    Alamine_aminotransferase =  st.number_input('Alamine_Aminotransferase')
    Aspartate_aminotransferase =  st.number_input('Aspartate_Aminotransferase')
    Total_protiens =  st.number_input('Total_Protiens')
    Albumin=  st.number_input('Albumin')
    Albumin_Globulin_Ratio =  st.number_input('Albumin_and_Globulin_Ratio')



    diagnosis = ''
    
    if(st.button('Liver disease test result')):
        diagnosis = liver_prediction_lr([age,gender,t_bilirubin,d_bilirubin,Alkaline_phosphotase,Alamine_aminotransferase,Aspartate_aminotransferase,Total_protiens,Albumin,Albumin_Globulin_Ratio])
        
    st.success(diagnosis)

if __name__== '__main__':
    main()




 







    
    