 # -*- coding: utf-8 -*-


import numpy as np
import os
import pickle
import streamlit as st

#path
main_file_path=os.path.realpath(__file__)
file_path = os.path.join(os.path.dirname(main_file_path), "./pickle_models/kidney_rf.pkl")

# loaded model
kidney_rf_loaded_model = pickle.load(open(file_path,'rb'))




# Function for predicting

def kidney_prediction_rf(input_data):
    
    #changing the input data as numpy array
    input_data_as_np_array = np.asarray(input_data)
    
    #reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_np_array.reshape(1,-1 )
    
    prediction = kidney_rf_loaded_model.predict(input_data_reshaped)
    print(prediction)
    
    if(prediction[0]==1):
        return 'The person might have chronic kidney disease'
    else:
        return 'The person is unlikely to have chronic kidney disease'

def main():
    
    # giving a title
    st.title('Kidney Disease Prediction')
    
    #input data

    Age = st.number_input('Age')
    BloodPressure =  st.number_input('Blood Pressure')
    SpecificGravity = st.number_input('Specific Gravity')
    Sugar =  st.number_input('Sugar')
    RedBloodCellIsOk =  st.checkbox('Red Blood Cell count is OK')
    PusCellsIsOk =  st.checkbox('Pus Cells are Normal')
    PusCellClumpIsPresent =  st.checkbox('Pus Cell Clumps are present')
    BacteriaIsPresent =  st.checkbox('Bacteria is Present')
    SerumCreatinine =  st.number_input(' Serum Creatinine')
    Sodium =  st.number_input('Sodium')
    Pottasium =  st.number_input('Pottasium')
    Hemoglobin =  st.number_input('Hemoglobin')
    PackedCellVolume =  st.number_input('Packed Cell Volume')
    WhiteBloodCellCount =  st.number_input('White Blood Cell Count')
    RedBloodCellCount =  st.number_input('Red Blood Cell Count')
    Hypertension =  st.checkbox('Hypertension')
    CoronaryArteryDisease =  st.checkbox('Coronary artery disease')
    Appetite =  st.checkbox('Appetite')
    PedalEdema =  st.checkbox('Pedal Edema')
    Anemia =  st.checkbox('Anemia')


    diagnosis = ''
    
    if(st.button('Kidney test result')):
        diagnosis = kidney_prediction_rf([Age,BloodPressure,SpecificGravity,Sugar,RedBloodCellIsOk,PusCellsIsOk,PusCellClumpIsPresent,BacteriaIsPresent,SerumCreatinine,Sodium,Pottasium,Hemoglobin,PackedCellVolume,WhiteBloodCellCount,RedBloodCellCount,Hypertension,CoronaryArteryDisease,Appetite,PedalEdema,Anemia])
        
    st.success(diagnosis)

if __name__== '__main__':
    main()




 







    
    