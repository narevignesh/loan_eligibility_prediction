import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

def load_and_train_model(file_path):
    data = pd.read_csv(file_path)
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
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    logistic_model = LogisticRegression(random_state=42)
    logistic_model.fit(X_train, y_train)
    log_accuracy = accuracy_score(y_test, logistic_model.predict(X_test))
    print("Logistic Regression Accuracy:", log_accuracy)
    pickle.dump(logistic_model, open("logistic_model.pkl", "wb"))
    pickle.dump(label_encoder, open("label_encoder.pkl", "wb"))
    pickle.dump(scaler, open("scaler.pkl", "wb"))
    pickle.dump(X_encoded.columns, open("feature_columns.pkl", "wb"))
    return "Logistic Regression model trained and saved successfully."

if __name__ == "__main__":
    load_and_train_model("train_loan.csv")