import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "customer_churn_sample.csv"
MODEL_PATH = BASE_DIR / "model.joblib"
COLS_PATH = BASE_DIR / "model.joblib.cols"

# Load data
df = pd.read_csv(DATA_PATH)

# -----------------------------
# CLEANING (IMPORTANT)
# -----------------------------
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna(subset=["TotalCharges"])

# Target
y = df["Churn"].map({"Yes": 1, "No": 0})

# Features
X = df[
    [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "Contract",
        "InternetService",
        "PaymentMethod",
    ]
]

# Column groups
num_features = ["tenure", "MonthlyCharges", "TotalCharges"]
cat_features = ["Contract", "InternetService", "PaymentMethod"]

# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
    ]
)

# Model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

# Pipeline
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ]
)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train
pipeline.fit(X_train, y_train)

# -----------------------------
# SAVE MODEL + COLUMNS
# -----------------------------
joblib.dump(pipeline, MODEL_PATH)

# Extract final feature names (CRITICAL for SHAP)
feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
joblib.dump(feature_names, COLS_PATH)

print("✅ Model trained & saved successfully")
print(f"📦 Total features after encoding: {len(feature_names)}")
