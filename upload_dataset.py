from app.dataset import upload_csv_to_mongo

count = upload_csv_to_mongo("data/customer_churn_sample.csv")
print(f"✅ {count} rows uploaded to MongoDB")
