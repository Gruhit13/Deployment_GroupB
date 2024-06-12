#!/usr/bin/env python
# coding: utf-8

# In[2]:


from flask import Flask, request, jsonify
from flask_wtf import CSRFProtect
import numpy as np
from model import load_model_and_categories, preprocess_image

app = Flask(__name__)
csrf = CSRFProtect(app)

# Load the model and categories
dnn_model, categories = load_model_and_categories()

@app.route('/')
def index():
    return "Welcome to the Lung Infection Prediction API!"

@app.route('/predict', methods=['POST'])
@csrf.exempt  # Disable CSRF protection for this route
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file provided'}), 400

    try:
        img = preprocess_image(file)
        dnn_prediction = dnn_model.predict(img)
        dnn_class = categories[np.argmax(dnn_prediction)]
        
        return jsonify({
            'dnn_prediction': dnn_class
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)


# In[ ]:




