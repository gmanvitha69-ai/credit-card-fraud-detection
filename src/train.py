import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load processed dataset
data = pd.read_csv("data/processed_data.csv")

print("Processed Dataset Loaded!")

# Select ONLY required columns
data = data[['Time', 'V1', 'V2', 'V3', 'Amount', 'Class']]

# Separate fraud and genuine
normal = data[data['Class'] == 0]
fraud = data[data['Class'] == 1]

# Balance dataset
sample_size = min(len(normal), len(fraud))

normal_sample = normal.sample(n=sample_size, random_state=42)
fraud_sample = fraud.sample(n=sample_size, random_state=42)

# Combine
new_data = pd.concat([normal_sample, fraud_sample], axis=0)

# Features and labels
X = new_data.drop(columns=['Class'])
Y = new_data['Class']

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=2
)

# Create model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train model
model.fit(X_train, Y_train)

# Predictions
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# Accuracy
train_accuracy = accuracy_score(Y_train, train_predictions)
test_accuracy = accuracy_score(Y_test, test_predictions)

print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)

# Save model
joblib.dump(model, "models/fraud_model.pkl")

print("Model Saved Successfully!")