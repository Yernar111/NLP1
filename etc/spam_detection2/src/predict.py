import joblib

pipeline = joblib.load('models/pipeline.pkl')

new_emails = [
    "Get rich quick! Click this link to claim $100000",
    "Hi Team, please find attached the meeting notes from today."
]

predictions = pipeline.predict(new_emails)
print(predictions)