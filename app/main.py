from fastapi import FastAPI, HTTPException
from app.schemas import ChurnInput
from app.model import load_model, prepare_input, get_shap_values
from app.dataset import save_prediction

app = FastAPI(title="Telecom Churn Prediction API")

# Load model ONCE
pipeline, preprocessor, model, feature_names, explainer = load_model()


@app.get("/")
def root():
    return {"message": "Telecom Churn Prediction API is running"}


# -------------------- PREDICTION -------------------- #
@app.post("/predict")
def predict(data: ChurnInput):
    X = prepare_input(data)

    prediction = int(pipeline.predict(X)[0])
    probability = float(pipeline.predict_proba(X)[0][1])

    result = {
        "churn_prediction": "Yes" if prediction == 1 else "No",
        "churn_probability": round(probability, 2)
    }

    save_prediction(
        input_data=data.dict(),
        prediction=result["churn_prediction"],
        probability=result["churn_probability"]
    )

    return result


# -------------------- EXPLANATION -------------------- #
@app.post("/explain")
def explain(data: ChurnInput):
    try:
        # Prepare input
        X = prepare_input(data)

        # Transform features
        X_transformed = preprocessor.transform(X)

        # Predict
        prediction = int(model.predict(X_transformed)[0])
        probability = float(model.predict_proba(X_transformed)[0][1])

        # SHAP values (SAFE)
        shap_vals = get_shap_values(explainer, X_transformed)

        # Pair with feature names
        pairs = list(zip(feature_names, shap_vals.tolist()))
        pairs.sort(key=lambda x: abs(x[1]), reverse=True)
        top_pairs = pairs[:5]

        explanation = [
            {
                "feature": feature,
                "impact": round((value), 3),
                "effect": "increases churn risk" if value > 0.0 else "reduces churn risk"
            }
            for feature, value in top_pairs
        ]

        return {
            "churn_prediction": "Yes" if prediction == 1 else "No",
            "churn_probability": round(probability, 2),
            "top_factors": explanation
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
