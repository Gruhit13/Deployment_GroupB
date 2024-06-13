import numpy as np
from PIL import Image
from io import BytesIO

MODEL_IMAGE_WIDTH = 256
MODEL_IMAGE_HEIGHT = 256

def load_image(img_data):
    image = Image.open(BytesIO(img_data))

    return image

def preprocess_image(image):
    # Resize the image to be of the model size
    image = image.resize((MODEL_IMAGE_WIDTH, MODEL_IMAGE_HEIGHT))

    # Convert it to grayscale if not
    image = image.convert('L')

    return image

def predict(image, model):
    # Convert the image to numpy array
    image = np.array(image)

    # Add an extra dimension at the end
    image = np.expand_dims(image, axis=-1)

    # Also add one dimension at the front ot make it as single batch
    batch_img = np.expand_dims(image, axis=0)

    print("Batch Image shape: ", batch_img.shape)

    # Make the prediction from the model
    pred_probs = model.predict(batch_img)[0]
    label = np.argmax(pred_probs, axis=-1)

    return {
        'pred_probs': pred_probs.tolist(),
        'label': int(label)
    }