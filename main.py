import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

data = pd.read_csv("data/smart_manufacturing_data.csv")

print("Dataset loaded successfully!")
print("Number of rows:", len(data))

# Use all important sensor features
features = [
    "temperature",
    "vibration",
    "humidity",
    "pressure",
    "energy_consumption"
]

X = data[features]
y_true = data["anomaly_flag"]

# Train Isolation Forest
model = IsolationForest(
    contamination=0.09,
    random_state=42
)

model.fit(X)

# Predict anomalies
predictions = model.predict(X)

data["predicted_anomaly"] = pd.Series(predictions).map({
    -1: 1,
    1: 0
})

y_pred = data["predicted_anomaly"]

# Results
print("\n--- Detection Results ---")
print("Normal readings:", (y_pred == 0).sum())
print("Anomalies detected:", (y_pred == 1).sum())
print("Actual anomalies:", y_true.sum())

# Evaluation
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print("\n--- Final Model Evaluation ---")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-Score : {f1:.4f}")