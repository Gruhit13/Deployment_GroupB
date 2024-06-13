import streamlit as st
import requests
import json
from io import BytesIO

st.title('COVID Prediction using DNN')

image_file = st.file_uploader("Upload Image")

if image_file is not None:

    image = image_file.getvalue()

    response = requests.post(
        "https://brijal08-covid19-dnn.hf.space/prediction",
        files = {
            "image": BytesIO(image)
        }
    )

    label = json.loads(response._content)
    st.write(f"Patient has {label['class']}")