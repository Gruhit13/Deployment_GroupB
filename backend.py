import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from fastapi import FastAPI, UploadFile, File
from utils import load_image, preprocess_image, predict
from model import get_model
import json

app = FastAPI()

MODEL_WEIGHT_PATH = './finetune_v1_weights.keras'
model = get_model(MODEL_WEIGHT_PATH)

@app.get("/")
def tester():
    return {
        "status": "Hello World"
    }

@app.post("/get_prediction")
async def get_prediction(x_ray_image: UploadFile = File(...)):

    # Load the image i.e. convert from bytes -> Image (unit8)
    image = load_image(await x_ray_image.read())

    # Preprocess image to make it compatible for model
    image = preprocess_image(image)

    # Retrive model prediction
    prediction = predict(image, model)

    print("Model Predicted: \n", prediction)

    return {
        'prediction': json.dumps(prediction)
    }

@app.post("/test")
def test():
    return {
        "status": 10
    }