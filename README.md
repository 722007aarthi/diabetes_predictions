#  Diabetes Risk Prediction System
 
A machine learning web application that predicts the risk of diabetes based on health parameters using the PIMA Indians Diabetes Dataset.
 
 
## Live Demo
> Run locally at http://127.0.0.1:5000/

 ## 📸 Screenshots
 <img width="1880" height="917" alt="image" src="https://github.com/user-attachments/assets/3910b6c2-060e-4e95-8848-7add7de2947a" />
 <img width="1891" height="915" alt="image" src="https://github.com/user-attachments/assets/b2036385-8e4c-4f9a-8f2d-22e4b7fc110f" />
 
 ##  About The Project
 
This project uses a **Random Forest Classifier** trained on the PIMA Indians Diabetes Dataset to predict whether a person is **Diabetic** or **Non-Diabetic** based on 8 medical input features.
 
Class imbalance was handled using **oversampling** technique to ensure the model correctly detects both diabetic and non-diabetic cases.

 
## Tech Stack
 
| Layer | Technology |
|-------|-----------|
| Language | Python 3 |
| ML Model | Random Forest (Scikit-learn) |
| Web Framework | Flask |
| Frontend | HTML, CSS, Bootstrap 5 |
| Dataset | PIMA Indians Diabetes Dataset |
 
 
## Model Performance
 
| Metric | Score |
|--------|-------|
| Accuracy | 88% |
| Precision | 0.88 |
| Recall | 0.88 |
| F1-Score | 0.88 |
 

 
## Project Structure

diabetes_predictions/
├── app.py                  # Flask web application
├── diabetes.py             # ML model training script
├── diabetes.csv            # Dataset
├── diabetes_model.pkl      # Trained model (generated)
├── requirements.txt        # Python dependencies
└── templates/
    ├── index.html          # Input form page
    └── result.html         # Prediction result page

##  How To Run
 
**1. Clone the repository**
git clone https://github.com/722007aarthi/diabetes_predictions.git
cd diabetes_predictions
 
**2. Install dependencies**
pip install -r requirements.txt
 
**3. Train the model**
python diabetes.py
 
**4. Run the app**
python app.py
 
**5. Open browser**
http://127.0.0.1:5000/


 
## Input Features
 
| Feature | Description |
|---------|-------------|
| Pregnancies | Number of times pregnant |
| Glucose | Plasma glucose concentration (mg/dL) |
| Blood Pressure | Diastolic blood pressure (mm Hg) |
| Skin Thickness | Triceps skin fold thickness (mm) |
| Insulin | 2-Hour serum insulin (μU/mL) |
| BMI | Body Mass Index (kg/m²) |
| Diabetes Pedigree Function | Family history score |
| Age | Age in years |
 
 
## How It Works
 
1. User enters health details in the form
2. Data is sent to Flask backend
3. Input is scaled using StandardScaler
4. Random Forest model predicts the outcome
5. Result is displayed — **Diabetic** or **Non-Diabetic**


