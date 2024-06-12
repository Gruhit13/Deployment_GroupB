#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

def load_model_and_categories():
    dnn_model = load_model('./model.h5')
    categories = ['covid-19', 'non-covid', 'normal']
    return dnn_model, categories

def preprocess_image(file):
    img = Image.open(file)
    img = img.convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array


# In[ ]:




