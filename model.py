import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

# Load dataset
data = pd.read_csv("dataset.csv")

# Encode crop names
le = LabelEncoder()
data["crop"] = le.fit_transform(data["crop"])

# Split features and target
X = data.drop("fertilizer", axis=1)
y = data["fertilizer"]

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model and encoder
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(le, open("encoder.pkl", "wb"))

print("Model trained and saved successfully")
