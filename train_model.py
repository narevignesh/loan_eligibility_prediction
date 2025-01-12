import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

# Load and preprocess the data
data = pd.read_csv("train_loan.csv")
data = data.drop(columns=["Loan_ID"])
X = data.drop(columns=["Loan_Status"])
y = data["Loan_Status"]

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

imputer = SimpleImputer(strategy="most_frequent")
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
X_encoded = pd.get_dummies(X_imputed, drop_first=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Train a Logistic Regression model
logistic_model = LogisticRegression(random_state=42, max_iter=1000)
logistic_model.fit(X_train, y_train)

# Save model and preprocessors
with open("model.pkl", "wb") as model_file:
    pickle.dump(logistic_model, model_file)

with open("scaler.pkl", "wb") as scaler_file:
    pickle.dump(scaler, scaler_file)

with open("label_encoder.pkl", "wb") as label_encoder_file:
    pickle.dump(label_encoder, label_encoder_file)

print("Model, scaler, and label encoder saved successfully!")
