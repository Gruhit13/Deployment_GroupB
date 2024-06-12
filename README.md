# Covid Predictor

This an application to predict the covid-19 disease that outburseted in 2020 and created a serious pendamic. 

Over here we created a neural network that is able to distinguish from a person's lungs x-rays if the patient has COVID19,
some other disease or is totally normal.

We used convolution neural network to extract spatial features from the image and created a vectorize representation of the
image that was futher provided to a fully connected network to provide the confidense for each of category. Thus, along
with just predicting the disease this model even provides its confidense in its prediction. 

### Frontend
Have a look at the application by clicking the link 👉: [Covid Predictor](https://covid-predictor.streamlit.app/)

### Backend
We used Huggingface's spaces to deploy the Neural Network and the link for it is 👉: [Backend](https://huggingface.co/spaces/gruhit-patel/covid_prediction/tree/main) <br>
Due to github's limitation on file size you can clone the hugging face repo to replicate the backend 

#### App Demo
[covid-predictor.webm](https://github.com/Gruhit13/Deployment_GroupB/assets/64111603/50c3460e-be1d-4a74-bc14-61577b1d7a09)

### COVID Predictor Backend

This repository contains the code for a COVID prediction model. Follow the steps below to clone the repository, install the required dependencies, and run the application.

## Cloning the Repository

1. Clone the repository using the following command:
   ```bash
   git clone https://huggingface.co/spaces/gruhit-patel/covid_prediction
   ```

2. Change to the repository directory:
   ```bash
   cd covid_prediction
   ```

## Installing Dependencies

1. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   ```

2. Activate the virtual environment:
   - On Windows:
     ```bash
     .\venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

1. Run the application using Uvicorn:
   ```bash
   uvicorn server:app --reload --port 3000
   ```

   This will start the server on port 3000.

2. Open your web browser and navigate to `http://127.0.0.1:3000` to access the application.
