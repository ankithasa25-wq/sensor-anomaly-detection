import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load dataset
data = pd.read_csv("data/smart_manufacturing_data.csv")

print("Dataset loaded successfully!")
print("Number of rows:", len(data))

# Sensor features used by the model
features = ["temperature", "vibration", "humidity"]

X = data[features]
y_true = data["anomaly_flag"]

# Create Isolation Forest model
model = IsolationForest(
    contamination=0.09,
    random_state=42
)

# Train model
model.fit(X)

# Predict anomalies
predictions = model.predict(X)

data["predicted_anomaly"] = pd.Series(predictions).map({
    -1: 1,
    1: 0
})

y_pred = data["predicted_anomaly"]

# Detection results
normal_count = (y_pred == 0).sum()
anomaly_count = (y_pred == 1).sum()
actual_anomalies = y_true.sum()

print("\n--- Detection Results ---")
print("Normal readings:", normal_count)
print("Anomalies detected:", anomaly_count)
print("Actual anomalies:", actual_anomalies)

# Model evaluation
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print("\n--- Model Evaluation ---")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-Score : {f1:.4f}")

# Separate normal and anomaly readings
normal_data = data[data["predicted_anomaly"] == 0]
anomaly_data = data[data["predicted_anomaly"] == 1]

# Create visualization
plt.figure(figsize=(10, 6))

plt.scatter(
    normal_data["temperature"],
    normal_data["vibration"],
    alpha=0.4,
    label="Normal"
)

plt.scatter(
    anomaly_data["temperature"],
    anomaly_data["vibration"],
    alpha=0.7,
    label="Anomaly"
)

plt.xlabel("Temperature")
plt.ylabel("Vibration")
plt.title("Sensor Anomaly Detection using Isolation Forest")
plt.legend()

# Save graph
plt.savefig("anomaly_detection_graph.png")

plt.show()