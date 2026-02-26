from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import pandas as pd
import webbrowser
from threading import Timer

# Initialize FastAPI app
app = FastAPI(title="Customer Churn Prediction API")

# Load the saved pipeline
model_pipeline = joblib.load('churn_model_pipeline.pkl')

# Define input data schema
class CustomerData(BaseModel):
    CreditScore: int
    Geography: str
    Gender: str
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float

# Custom UI Homepage
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Customer Churn Predictor</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }
            .container {
                max-width: 800px;
                margin: 0 auto;
                background: white;
                padding: 40px;
                border-radius: 20px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            }
            h1 {
                color: #667eea;
                text-align: center;
                margin-bottom: 10px;
                font-size: 2.5em;
            }
            .subtitle {
                text-align: center;
                color: #666;
                margin-bottom: 30px;
            }
            .form-grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
                margin-bottom: 20px;
            }
            .form-group {
                display: flex;
                flex-direction: column;
            }
            label {
                font-weight: 600;
                color: #333;
                margin-bottom: 8px;
                font-size: 14px;
            }
            input, select {
                padding: 12px;
                border: 2px solid #e0e0e0;
                border-radius: 8px;
                font-size: 14px;
                transition: border-color 0.3s;
            }
            input:focus, select:focus {
                outline: none;
                border-color: #667eea;
            }
            .btn {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 15px;
                border: none;
                border-radius: 8px;
                font-size: 16px;
                font-weight: bold;
                cursor: pointer;
                width: 100%;
                margin-top: 20px;
                transition: transform 0.2s;
            }
            .btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
            }
            .result {
                margin-top: 30px;
                padding: 25px;
                border-radius: 12px;
                display: none;
                text-align: center;
            }
            .result.show { display: block; }
            .result.high { background: #fee; border-left: 5px solid #f44336; }
            .result.medium { background: #fff3cd; border-left: 5px solid #ff9800; }
            .result.low { background: #e8f5e9; border-left: 5px solid #4caf50; }
            .result h2 { margin-bottom: 15px; }
            .probability {
                font-size: 3em;
                font-weight: bold;
                margin: 15px 0;
            }
            .loading {
                display: none;
                text-align: center;
                margin-top: 20px;
            }
            .loading.show { display: block; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔮 Customer Churn Predictor</h1>
            <p class="subtitle">AI-Powered Customer Retention Analysis</p>
            
            <form id="churnForm">
                <div class="form-grid">
                    <div class="form-group">
                        <label>Credit Score</label>
                        <input type="number" id="CreditScore" required min="300" max="850">
                    </div>
                    <div class="form-group">
                        <label>Geography</label>
                        <select id="Geography" required>
                            <option value="">Select...</option>
                            <option value="France">France</option>
                            <option value="Germany">Germany</option>
                            <option value="Spain">Spain</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Gender</label>
                        <select id="Gender" required>
                            <option value="">Select...</option>
                            <option value="Male">Male</option>
                            <option value="Female">Female</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Age</label>
                        <input type="number" id="Age" required min="18" max="100">
                    </div>
                    <div class="form-group">
                        <label>Tenure (Years)</label>
                        <input type="number" id="Tenure" required min="0" max="10">
                    </div>
                    <div class="form-group">
                        <label>Balance ($)</label>
                        <input type="number" id="Balance" required step="0.01">
                    </div>
                    <div class="form-group">
                        <label>Number of Products</label>
                        <input type="number" id="NumOfProducts" required min="1" max="4">
                    </div>
                    <div class="form-group">
                        <label>Has Credit Card</label>
                        <select id="HasCrCard" required>
                            <option value="">Select...</option>
                            <option value="1">Yes</option>
                            <option value="0">No</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Is Active Member</label>
                        <select id="IsActiveMember" required>
                            <option value="">Select...</option>
                            <option value="1">Yes</option>
                            <option value="0">No</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Estimated Salary ($)</label>
                        <input type="number" id="EstimatedSalary" required step="0.01">
                    </div>
                </div>
                <button type="submit" class="btn">🚀 Predict Churn</button>
            </form>

            <div class="loading" id="loading">⏳ Analyzing customer data...</div>
            
            <div class="result" id="result">
                <h2 id="resultTitle"></h2>
                <div class="probability" id="probability"></div>
                <p id="resultText"></p>
            </div>
        </div>

        <script>
            document.getElementById('churnForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                
                const loading = document.getElementById('loading');
                const result = document.getElementById('result');
                result.classList.remove('show');
                loading.classList.add('show');
                
                const data = {
                    CreditScore: parseInt(document.getElementById('CreditScore').value),
                    Geography: document.getElementById('Geography').value,
                    Gender: document.getElementById('Gender').value,
                    Age: parseInt(document.getElementById('Age').value),
                    Tenure: parseInt(document.getElementById('Tenure').value),
                    Balance: parseFloat(document.getElementById('Balance').value),
                    NumOfProducts: parseInt(document.getElementById('NumOfProducts').value),
                    HasCrCard: parseInt(document.getElementById('HasCrCard').value),
                    IsActiveMember: parseInt(document.getElementById('IsActiveMember').value),
                    EstimatedSalary: parseFloat(document.getElementById('EstimatedSalary').value)
                };

                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(data)
                    });
                    
                    const prediction = await response.json();
                    loading.classList.remove('show');
                    
                    result.className = 'result show ' + prediction.risk_level.toLowerCase();
                    document.getElementById('resultTitle').textContent = 
                        prediction.prediction === 1 ? '⚠️ High Churn Risk!' : '✅ Low Churn Risk';
                    document.getElementById('probability').textContent = 
                        (prediction.churn_probability * 100).toFixed(1) + '%';
                    document.getElementById('resultText').textContent = 
                        `Risk Level: ${prediction.risk_level}`;
                        
                } catch (error) {
                    loading.classList.remove('show');
                    alert('Error: ' + error.message);
                }
            });
        </script>
    </body>
    </html>
    """

@app.post("/predict")
def predict(customer: CustomerData):
    try:
        input_dict = customer.dict()
        input_df = pd.DataFrame([input_dict])
        
        prediction = model_pipeline.predict(input_df)[0]
        probability = model_pipeline.predict_proba(input_df)[0][1]
        
        if probability < 0.3:
            risk_level = "Low"
        elif probability < 0.6:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        return {
            'prediction': int(prediction),
            'churn_probability': round(float(probability), 4),
            'risk_level': risk_level
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Auto-open browser
def open_browser():
    webbrowser.open_new("http://127.0.0.1:8000")

@app.on_event("startup")
async def startup_event():
    Timer(1.5, open_browser).start()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)