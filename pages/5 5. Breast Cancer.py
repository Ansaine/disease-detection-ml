 # -*- coding: utf-8 -*-


import numpy as np
import os
import pickle
import streamlit as st

#path
main_file_path=os.path.realpath(__file__)
file_path = os.path.join(os.path.dirname(main_file_path), "./pickle_models/bcancer_lr.pkl")

# loaded model
bcancer_lr_loaded_model = pickle.load(open(file_path,'rb'))




# Function for predicting

def bcancer_prediction_lr(input_data):
    
    #changing the input data as numpy array
    input_data_as_np_array = np.asarray(input_data)
    
    #reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_np_array.reshape(1,-1 )
    
    prediction = bcancer_lr_loaded_model.predict(input_data_reshaped)
    print(prediction)
    
    if(prediction[0]==1):
        return 'The person might have Breast Cancer'
    else:
        return 'The The person does not have Breast Cancer'

def main():
    
    # giving a title
    st.title('Breast Cancer Prediction')
    st.write('Using Logistic Regression algorithm')
    
    #input data

    radius_mean = st.number_input('Mean Radius')
    texture_mean =  st.number_input('Mean Texture')
    smoothness_mean =  st.number_input('Mean smoothness')
    symmetry_mean =  st.number_input('Symmetry mean')
    fractal =  st.number_input('Fractal Dimension mean')
    radius_se =  st.number_input('Standard deviation of Radius')
    texture_se=  st.number_input('Standard deviation of Texture')
    smoothness_se =  st.number_input('Standard deviation of smoothness')
    compactness_se =  st.number_input('Standard deviation of compactness')
    concavity_se =  st.number_input('Standard deviation of concavity')
    concave_se =  st.number_input('Standard deviation of concave points')
    symmetry_se =  st.number_input('Standard deviation of symmetry')
    smoothness_worst =  st.number_input('Smoothness worst')
    concavity_worst =  st.number_input('Concavity worst')
    symmetry_worst =  st.number_input('Symmetry worst')
    fractal_worst =  st.number_input('Fractal Dimension worst')
    

    diagnosis = ''
    
    if(st.button('Breast Cancer test result')):
        diagnosis = bcancer_prediction_lr([radius_mean,texture_mean,smoothness_mean,symmetry_mean,fractal,radius_se,texture_se,smoothness_se,compactness_se,concavity_se,concave_se,symmetry_se,smoothness_worst,concavity_worst,symmetry_worst,fractal_worst])
        
    st.success(diagnosis)

if __name__== '__main__':
    main()




 







    
    