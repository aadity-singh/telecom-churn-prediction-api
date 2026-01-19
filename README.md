📞 Telecom Churn Prediction API
   Machine Learning–Powered REST API to Predict Customer Churn

🚀 Project Overview
   Customer churn is one of the biggest challenges for telecom companies.
   This project delivers a production-ready Machine Learning API that predicts whether a customer is likely to churn based on their usage patterns, contract details, and service preferences.
   The solution follows industry-standard ML engineering practices, including data preprocessing pipelines, model serialization, explainability (SHAP), and API deployment readiness.

🎯 Key Objectives
    Predict customer churn (Yes / No) with high reliability
    Provide a scalable REST API for real-time predictions
    Maintain clean, modular ML architecture
    Enable model interpretability for business insights.

🧠 Machine Learning Approach
    Problem Type: Binary Classification
    Target Variable: Churn
    Model Pipeline:
      Data Cleaning & Feature Engineering
      Encoding & Scaling
      Trained ML Model (serialized)
      SHAP for explainability

🛠️ Tech Stack
    Category	                   Tools
    Language	                   Python
    API Framework              	 FastAPI
    ML Libraries	             scikit-learn
    Explainability	             SHAP
    Data Handling	             Pandas, NumPy
    Model Serving	               Uvicorn
    Version Control	           Git & GitHub
    Deployment Ready	         Render / Cloud

📁 Project Structure
    telecom-churn-prediction-api/
    │
    ├── app/
    │   ├── main.py              # FastAPI app entry point
    │
    ├── pipeline/
    │   ├── preprocessing.py     # Feature engineering & transformations
    │
    ├── models/
    │   ├── churn_model.pkl      # Trained ML model
    │
    ├── shap/
    │   ├── explainer.pkl        # SHAP explainer
    │
    ├── requirements.txt
    ├── README.md


🔌 API Endpoints
 ➤ Health Check:
   GET /
   
 #Response

    {
      "status": "API is running successfully"
    }

 ➤ Predict Churn
  POST /predict

  #Sample Request

    {
      "gender": "Male",
      "SeniorCitizen": 0,
      "Partner": "Yes",
      "Dependents": "No",
      "tenure": 12,
      "PhoneService": "Yes",
      "InternetService": "Fiber optic",
      "MonthlyCharges": 85.5,
      "TotalCharges": 1025.3
    }

 #Sample Response

    {
      "churn_prediction": "Yes",
      "probability": 0.82
    }

 📊 Model Explainability (SHAP)
    This project integrates SHAP (SHapley Additive Explanations) to:
    Understand feature impact on churn predictions
    Improve transparency for business stakeholders
    Support data-driven retention strategies

  ▶️ Run Locally
  1️⃣ Clone the repository
  git clone https://github.com/aadity-singh/telecom-churn-prediction-api.git
  cd telecom-churn-prediction-api
  
  2️⃣ Create virtual environment
  python -m venv venv
  venv\Scripts\activate   # Windows
  
  3️⃣ Install dependencies
  pip install -r requirements.txt
  
  4️⃣ Start the API
  uvicorn app.main:app --reload
  
  API will run at:
  http://127.0.0.1:8000
  
  Swagger Docs:
  http://127.0.0.1:8000/docs

⭐ Why This Project Stands Out
    Unlike basic ML notebooks or demo apps, this project is built with a production-first mindset:
    End-to-End ML Pipeline – from preprocessing to prediction via a live API
    Real Business Problem – telecom churn directly impacts revenue & retention
    API-First Design – ready to integrate with CRM or web applications
    Model Explainability (SHAP) – adds transparency and trust to predictions
    Clean Project Structure – follows industry-level ML engineering standards
    Deployment Ready – structured for cloud hosting (Render / similar platforms)
📌 This project demonstrates both Data Science and Machine Learning Engineering skills, making it suitable for Data Scientist, ML Engineer, and Analyst roles.

🔮 Next Enhancements (Planned Improvements)
    To further improve scalability, reliability, and real-world usability, the following enhancements are planned:
    🔐 Authentication & Authorization
    Secure the API using API keys or OAuth
    📦 Batch Prediction Endpoint
    Enable predictions for multiple customers at once
    📈 Model Monitoring & Drift Detection
    Track performance degradation over time
    🧪 Automated Testing
    Unit and integration tests for API reliability
    ⚙️ CI/CD Pipeline
    Automatic testing and deployment on every update
    📊 Interactive Dashboard
    Visualize churn trends and feature importance
    
  🚧 Deployment:
    🚀 Render Deployment: Coming Soon
    (The project is fully cloud-deployment ready.)
    
  📌 Use Cases:
      Customer retention strategy
      CRM system integration
      Telecom business analytics
      ML API portfolio demonstration

 🧑‍💻 Author

    Aadity Singh
    Aspiring Data Scientist | Machine Learning Engineer
    🔗 GitHub: https://github.com/aadity-singh
