import pandas as pd
from app.database import collection
from datetime import datetime
def upload_csv_to_mongo(csv_path: str):
    df = pd.read_csv(csv_path)
    records = df.to_dict(orient="records")

    collection.delete_many({})  # avoid duplicates
    collection.insert_many(records)

    return len(records)

def get_dataset():
    data = list(collection.find({}, {"_id": 0}))
    return pd.DataFrame(data)

def save_prediction(input_data, prediction, probability, explanation=None):
    doc = {
        "input": input_data,
        "prediction": prediction,
        "probability": probability,
        "explanation": explanation,
        "created_at": datetime.utcnow()
    }

    collection.insert_one(doc)