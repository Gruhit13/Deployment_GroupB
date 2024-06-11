# Bank Churn Prediction App

## Overview

This Bank Churn Prediction app leverages a deep learning model to predict the likelihood of customer churn for a bank. Churn prediction is critical for businesses to retain their customers by identifying those who are likely to leave. This app utilizes advanced machine learning techniques and a robust neural network to provide accurate churn predictions and insights.

## Features

### Data Preprocessing
- **Handling Missing Values**: Columns with a high percentage of missing values are removed, and the remaining missing values are imputed using appropriate strategies.
- **Categorical and Numerical Features**: Separate pipelines handle the preprocessing of categorical and numerical features, including scaling and encoding.

### Model Training
- **Deep Learning Network**: A sequential neural network with multiple dense layers, batch normalization, and dropout layers is used to train the model. The model is optimized using the Adam optimizer and binary cross-entropy loss function.
- **SMOTE**: Synthetic Minority Over-sampling Technique (SMOTE) is applied to handle class imbalance in the training data.
- **Callbacks**: Early stopping and learning rate reduction are implemented to enhance the training process.

### Model Evaluation
- **Performance Metrics**: The model's performance is evaluated using metrics such as accuracy, F1 score, ROC AUC, and confusion matrices for both training and validation sets.
- **PCA Visualization**: Principal Component Analysis (PCA) is used to visualize the variance explained by different components in the dataset.

### Streamlit App Interface
- **Customer Details**: Users can select a customer ID to view detailed information about the selected customer.
- **Churn Prediction**: The app predicts the churn probability for the selected customer, displaying the result in a user-friendly manner.
- **LIME Explanation**: Local Interpretable Model-agnostic Explanations (LIME) are used to explain the model's prediction, providing insights into the features that contributed to the churn prediction.

## Usage

1. **Data Upload**: Load the CSV file containing customer data.
2. **Select Customer**: Choose a customer ID from the dropdown list to view their details.
3. **Predict Churn**: Click the "Predict Churn" button to get the churn probability for the selected customer.
4. **Feature Importance**: View a bar plot explaining the feature importance and contribution to the churn prediction using LIME.

## Demo video
https://www.loom.com/share/e42abde209c34a0daf3deea85550da03?sid=0559de07-2a3c-4c90-95c8-117e6ba33641

## Technologies Used

- **Python Libraries**: pandas, numpy, seaborn, scikit-learn, imbalanced-learn, tensorflow, keras, joblib, lime
- **Deep Learning Framework**: TensorFlow and Keras
- **Web Framework**: Streamlit for creating the interactive web interface

## Conclusion

This Bank Churn Prediction app provides a comprehensive solution for predicting customer churn using a deep learning model. It offers an intuitive interface for users to interact with the model, visualize predictions, and understand the factors influencing customer churn. This tool can be instrumental for banks and financial institutions in strategizing customer retention efforts.
