import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import pickle
 
df = pd.read_csv("diabetes.csv")
 
# Replace invalid 0s with median for clinical columns
for col in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
    df[col] = df[col].replace(0, df[col].median())
 
X = df.drop("Outcome", axis=1)
y = df["Outcome"]
 
# Fix class imbalance — oversample diabetic cases
df_full = pd.concat([X, y], axis=1)
df_majority = df_full[df_full.Outcome == 0]
df_minority = df_full[df_full.Outcome == 1]
df_minority_upsampled = resample(df_minority, replace=True,
                                  n_samples=len(df_majority), random_state=42)
df_balanced = pd.concat([df_majority, df_minority_upsampled])
 
print("Balanced counts:", df_balanced["Outcome"].value_counts().to_dict())
 
X_bal = df_balanced.drop("Outcome", axis=1)
y_bal = df_balanced["Outcome"]
 
X_train, X_test, y_train, y_test = train_test_split(
    X_bal, y_bal, test_size=0.2, random_state=42
)
 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)
 
model = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
model.fit(X_train, y_train)
 
pred = model.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, pred) * 100, 2), "%")
print(classification_report(y_test, pred, target_names=["Non-Diabetic", "Diabetic"]))
 
# Save model + scaler together
pickle.dump({"model": model, "scaler": scaler}, open("diabetes_model.pkl", "wb"))
print("Model saved as diabetes_model.pkl")
 