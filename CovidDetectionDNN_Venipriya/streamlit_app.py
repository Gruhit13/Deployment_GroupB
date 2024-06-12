#!/usr/bin/env python
# coding: utf-8

# In[4]:


import streamlit as st
import requests

st.title("Lung Infection Prediction")

uploaded_file = st.file_uploader("Choose a lung X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
    
    st.write("")
    st.write("Classifying...")
    
    # Send the image to the Flask API for prediction
    files = {'file': uploaded_file.getvalue()}
    response = requests.post("http://127.0.0.1:5000/predict", files=files)
    
    if response.status_code == 200:
        predictions = response.json()
        st.write(f"DNN Model Prediction: {predictions['dnn_prediction']}")
    else:
        st.write(f"Error: {response.json()['error']}")


# In[ ]:




