🏦 Customer Churn Prediction - End-to-End ML Project

Python FastAPI scikit-learn License

📌 Project Overview
A complete machine learning solution to predict customer churn for a bank. This project includes data analysis, model training, hyperparameter tuning, and a production-ready web application built with FastAPI.

🎯 Model Performance:

ROC-AUC Score: 0.87
Accuracy: 0.84
Algorithm: Random Forest (after comparing with Logistic Regression)
🚀 Features
✅ End-to-End ML Pipeline - Data preprocessing → Training → Deployment
✅ Interactive Web UI - User-friendly interface for predictions
✅ REST API - FastAPI backend for integration
✅ Model Serialization - Saved using pickle for production use
✅ Hyperparameter Tuning - Optimized using GridSearchCV

🛠️ Tech Stack
Language: Python 3.8+
ML Libraries: scikit-learn, pandas, numpy
Web Framework: FastAPI
Server: Uvicorn
Data Visualization: Matplotlib, Seaborn (in notebook)
📁 Project Structure
Customer-Churn-Prediction/
│
├── data/
│ └── Churn_Modelling.csv # Dataset
│
├── notebooks/
│ └── Bank.ipynb # EDA + Model Training
│
├── models/
│ └── churn_model_pipeline.pkl # Trained model
│
├── app/
│ └── app.py # FastAPI application
│
├── screenshots/
│ ├── screenshot1.png # UI screenshots
│ └── screenshot2.png
│
├── requirements.txt # Dependencies
├── .gitignore # Git ignore file
└── README.md # Project documentation



---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
2️⃣ Create Virtual Environment (Recommended)

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
3️⃣ Install Dependencies

pip install -r requirements.txt
4️⃣ Run the Application

python app/app.py
Or:


uvicorn app.app:app --reload
5️⃣ Open in Browser

http://127.0.0.1:8000
📊 Model Development Process
1. Exploratory Data Analysis (EDA)
Analyzed customer demographics and banking behavior
Identified key features: Age, Balance, NumOfProducts, IsActiveMember
Handled missing values and outliers
2. Feature Engineering
Encoded categorical variables (Geography, Gender)
Scaled numerical features using StandardScaler
Created feature pipeline for reproducibility
3. Model Training & Comparison
Logistic Regression: Baseline model
Random Forest: Best performing (ROC-AUC: 0.87)
Used GridSearchCV for hyperparameter tuning
4. Model Evaluation
Confusion Matrix
ROC-AUC Curve
Precision, Recall, F1-Score
📸 Screenshots
Web Interface
Input Form

Prediction Result
Prediction

🔮 API Endpoints
GET /
Returns the web UI

POST /predict
Request Body:


{
  "CreditScore": 650,
  "Geography": "France",
  "Gender": "Male",
  "Age": 35,
  "Tenure": 5,
  "Balance": 50000,
  "NumOfProducts": 2,
  "HasCrCard": 1,
  "IsActiveMember": 1,
  "EstimatedSalary": 60000
}
Response:


{
  "prediction": "Churn",
  "churn_probability": 0.75
}
🎯 Future Improvements
 Deploy to cloud (AWS/Heroku/Render)
 Add model monitoring & retraining pipeline
 Implement A/B testing framework
 Create Docker container
 Add unit tests
📧 Contact
David Joe
📧 Email: Djosephus123@gmail.com


📄 License
This project is licensed under the MIT License.

⭐ If you found this project helpful, please give it a star! ⭐