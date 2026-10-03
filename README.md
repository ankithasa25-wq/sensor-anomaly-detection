# Sensor Anomaly Detection System

A Python-based machine learning system that detects unusual sensor readings in smart manufacturing data using the **Isolation Forest** algorithm.

## 📌 Project Overview

This project analyzes sensor data from a smart manufacturing environment and identifies unusual readings that may indicate abnormal machine behavior.

The machine learning model uses three sensor parameters:

* Temperature
* Vibration
* Humidity

The current software version processes a dataset containing **100,000 sensor records**.

## 🎯 Project Aim

To develop a sensor anomaly detection system that uses **Machine Learning to identify unusual operating conditions in machines**.

The current version works with stored sensor data. The planned future version will integrate real-time sensors and an ESP32 microcontroller.

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* PyCharm
* Isolation Forest

## 🤖 Machine Learning Model

The project uses the **Isolation Forest** algorithm, an unsupervised machine learning technique for detecting unusual or isolated data points.

The model analyzes:

* Temperature
* Vibration
* Humidity

Each sensor reading is classified as either:

* **Normal**
* **Anomaly**

The model uses a contamination value of **0.09**.

## 📊 Dataset

The dataset contains **100,000 smart manufacturing sensor records**.

### Features Used

| Feature     | Description                    |
| ----------- | ------------------------------ |
| Temperature | Machine temperature reading    |
| Vibration   | Machine vibration reading      |
| Humidity    | Environmental humidity reading |

The dataset contains:

* **91,084 actual normal readings**
* **8,916 actual anomaly readings**

## 📈 Detection Results

The trained Isolation Forest model produced the following results:

| Result                     |  Count |
| -------------------------- | -----: |
| Actual Normal Readings     | 91,084 |
| Actual Anomaly Readings    |  8,916 |
| Predicted Normal Readings  | 91,000 |
| Predicted Anomaly Readings |  9,000 |

## 🔍 Confusion Matrix

The confusion matrix obtained from the model was:

```text
[[85811, 5273],
 [ 5189, 3727]]
```

### Classification Breakdown

| Classification      |  Count | Meaning                                             |
| ------------------- | -----: | --------------------------------------------------- |
| True Negative (TN)  | 85,811 | Normal readings correctly identified as normal      |
| False Positive (FP) |  5,273 | Normal readings incorrectly identified as anomalies |
| False Negative (FN) |  5,189 | Anomalies incorrectly identified as normal          |
| True Positive (TP)  |  3,727 | Anomalies correctly identified as anomalies         |

## 📋 Model Evaluation

The model was evaluated using accuracy, precision, recall, and F1-score.

| Metric    | Result |
| --------- | -----: |
| Accuracy  | 89.54% |
| Precision | 41.41% |
| Recall    | 41.80% |
| F1-Score  | 41.61% |

## 📉 Visualization

The project generates a scatter plot showing the detected normal and anomalous sensor readings based on:

* Temperature
* Vibration

The generated graph is saved as:

`anomaly_detection_graph.png`

## 📁 Project Structure

```text
Sensor_Anomaly_Detection/
│
├── data/
│   └── smart_manufacturing_data.csv
│
├── main.py
├── anomaly_detection.py
├── anomaly_detection_graph.png
└── README.md
```

## ⚙️ How It Works

```text
Smart Manufacturing Dataset
          ↓
     Load Sensor Data
          ↓
Select Temperature, Vibration & Humidity
          ↓
    Isolation Forest
          ↓
   Anomaly Prediction
          ↓
 Normal / Anomaly
          ↓
Evaluation + Visualization
```

## 🚀 Future Development

The current version is a **software-based machine learning system**.

The planned next stage is to integrate real sensors with an **ESP32 microcontroller** so the system can process real-time sensor readings.

### Planned Embedded Architecture

```text
Temperature Sensor
        +
Vibration Sensor
        +
Humidity Sensor
        ↓
      ESP32
        ↓
Real-Time Sensor Data
        ↓
Python + Machine Learning
        ↓
   Isolation Forest
        ↓
  Normal / Anomaly
        ↓
Alert / Monitoring
```

The final system will combine **embedded hardware, real-time sensor data, Python, and machine learning** to detect unusual machine operating conditions.

## 📌 Current Status

* ✅ Dataset processing completed
* ✅ Isolation Forest model implemented
* ✅ Anomaly detection completed
* ✅ Model evaluation completed
* ✅ Visualization generated
* ✅ Python project uploaded to GitHub
* 🔄 ESP32 and real-sensor integration planned
