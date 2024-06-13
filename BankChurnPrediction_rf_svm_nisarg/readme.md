# Bank Churn Prediction App
### Overview
The Bank Churn Prediction app utilizes machine learning models, including Random Forest (RF), Support Vector Machine (SVM), and a deep learning neural network, to predict the likelihood of customer churn for a bank. Churn prediction is critical for businesses to retain their customers by identifying those who are likely to leave. This app leverages advanced machine learning techniques to provide accurate churn predictions and insights.

## Features
### Data Preprocessing
* **Handling Missing Values**: Columns with a high percentage of missing values are removed, and the remaining missing values are imputed using appropriate strategies.
* **Categorical and Numerical Features:** Separate pipelines handle the preprocessing of categorical and numerical features, including scaling and encoding.
## Model Training
* **Deep Learning Network**: A sequential neural network with multiple dense layers, batch normalization, and dropout layers is used to train the model. The model is optimized using the Adam optimizer and binary cross-entropy loss function.
* **SMOTE**: Synthetic Minority Over-sampling Technique (SMOTE) is applied to handle class imbalance in the training data.
* **Random Forest & SVM**: These traditional machine learning models are also trained and optimized for predicting customer churn.
* **Callbacks**: Early stopping and learning rate reduction are implemented to enhance the training process.

## Model Evaluation
* **Performance Metrics**: The model's performance is evaluated using metrics such as accuracy, F1 score, ROC AUC, and confusion matrices for both training and validation sets.
## Streamlit App Interface
* **Customer Details**: Users can select a customer ID to view detailed information about the selected customer.
* **Churn Prediction**: The app predicts the churn probability for the selected customer, displaying the result in a user-friendly manner.
* **LIME Explanation**: Local Interpretable Model-agnostic Explanations (LIME) are used to explain the model's prediction, providing insights into the features that contributed to the churn prediction.

## Usage
* Data Upload: Load the CSV file containing customer data.
* Select Customer: Choose a customer ID from the dropdown list to view their details.
* Predict Churn: Click the "Predict Churn" button to get the churn probability for the selected customer.
* Feature Importance: View a bar plot explaining the feature importance and contribution to the churn prediction using LIME.

## Technologies Used
* Python Libraries: pandas, numpy, seaborn, scikit-learn, imbalanced-learn, tensorflow, keras, joblib, lime
* Deep Learning Framework: TensorFlow and Keras
* Web Framework: Streamlit for creating the interactive web interface

## Conclusion
- This Bank Churn Prediction app provides a comprehensive solution for predicting customer churn using deep learning, Random Forest, and SVM models. It offers an intuitive interface for users to interact with the models, visualize predictions, and understand the factors influencing customer churn. This tool can be instrumental for banks and financial institutions in strategizing customer retention efforts.
