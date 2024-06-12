# Lung Infection Prediction API

This project provides a Flask-based API for predicting lung infections, specifically COVID-19, other infections, and no lung infection, using deep learning models.

## Directory Structure

Ensure your project directory has the following structure:


## Setup Instructions

Follow these steps to set up and run the Flask application using Waitress:

### 1. Install Required Packages

Ensure you have the required Python packages installed. You can install them using `pip`:
pip install flask numpy tensorflow waitress

### 2. Create minimal_app.py
Create a file named minimal_app.py

### 3. Verify Directory Structure
Ensure your directory structure

### 4. Navigate to Project Directory
Open a command prompt and navigate to your project directory:
cd C:\Users\venip\Documents\Course\Sem3\DL\assign3\projects

### 5. Install Waitress
Ensure you have Waitress installed:
pip install waitress

### 6. Run the Flask Application
Run the Waitress command to serve the app.py
waitress-serve --listen=0.0.0.0:5000 app:app

If the command runs without errors, you should be able to access the app at http://localhost:5000

## Troubleshooting
Ensure that app.py and model.py are in the same directory.
Ensure the Flask app object in app.py is named app.
Ensure there are no syntax errors or import issues in app.py and model.py.
If you encounter specific errors, please provide the error messages for further assistance.

## Summary
Install required packages.
Create a minimal Flask app to verify the setup.
Ensure directory structure is correct.
Create and run your main application using Waitress.

By following these steps, you should be able to set up and run your Flask application using Waitress on a Windows system.
