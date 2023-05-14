import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Health App"    
)

# Loading assets
img = Image.open("images/disease.jpg")

# Header Part
with st.container():
    st.subheader('B. Tech Final Year Project')
    st.text('Under supervision of : \nDr Madhuchanda Choudhury, Associate Professor, ECE \nNIT Silchar')
    st.title("Disease detection using multiple ML models")
    st.sidebar.success("Select a Disease")
    st.write("---")
    
# About section
with st.container():
    st.write("##")
    st.header("About : ")
    text_column, image_column = st.columns((2,1))
    with text_column:
        st.write("This Project focuses on comparative analysis of multiple machine learning algorithms for disease prediction and making an interactive website based on the most efficient algorithm to deliver all our learnings. The machine learning models were are Logistic Regression, Support Vector Machine(SVM) and Random Forest Classifier.")
        st.write("It can detect chanches of getting five types of diseases : ")
        st.write(" - Heart Disease")
        st.write(" - Chronic Kidney Disease")
        st.write(" - Diabetes")
        st.write(" - Liver Disease")
        st.write(" - Breast cancer")
    with image_column:
        st.image(img)
    st.write("---")
    
with st.container():
    st.subheader("Group Members")
    st.text("Rishav Saha           - 1914177\nAngshuman Das         - 1914116\nMukib Khan            - 1914034\nKamakhya Aareyo Deori – 1914002\nArun Kumar Kondapally – 1914140")

