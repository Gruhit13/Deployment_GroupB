import streamlit as st
import requests
import json
from io import BytesIO

st.title('COVID-19 Prediction')
st.subheader('Provide an X-ray of lungs and check if you have covid-19, or any other desease or your normal.')

# Take image input
img_file_buffer = st.file_uploader("Upload your chest X-ray")

class_labels = {
    0: 'COVID-19',
    1: 'Non-COVID',
    2: 'Normal'
}

if img_file_buffer is not None:
    # Retrive the file value
    img_bytes = img_file_buffer.getvalue()

    # Create a payload
    files = {"x_ray_image": BytesIO(img_bytes)}

    URL = "https://gruhit-patel-covid-prediction.hf.space/get_prediction"

    with st.spinner("Waiting for model response..."):
        st.write("Came in spinner")
        resp = requests.post(
            URL, 
            files = files,
            )

        if resp._content is not None:
            resp_data = json.loads(resp._content.decode('utf-8'))
            prediction = json.loads(resp_data['prediction'])
            
            label = prediction['label']
            confidense = prediction['pred_probs'][label]
            st.markdown(f"### Model Prediction: {class_labels[prediction['label']]} with {confidense*100:.1f} % confidense")
        else:
            st.write(resp.__dict__)


