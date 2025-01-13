---

## Model Description  
This project leverages a **Random Forest Classifier** to predict the approval status of loan applications based on customer data. The model was trained using a dataset of loan records with features related to applicant demographics, financial information, and loan-related attributes.  

### Features  
The model uses the following features:  
- **Gender**: Gender of the applicant (e.g., Male, Female).  
- **Married**: Marital status of the applicant (e.g., Yes, No).  
- **Dependents**: Number of dependents the applicant supports (e.g., 0, 1, 2, 3+).  
- **Education**: Applicant's education level (e.g., Graduate, Not Graduate).  
- **Self_Employed**: Employment status of the applicant (e.g., Yes, No).  
- **ApplicantIncome**: Applicant's monthly income in USD.  
- **CoapplicantIncome**: Co-applicant's monthly income in USD.  
- **LoanAmount**: Loan amount requested by the applicant (in thousands).  
- **Loan_Amount_Term**: Loan repayment term in months.  
- **Credit_History**: Binary representation of the applicant's credit history (1.0 = Good, 0.0 = Bad).  
- **Property_Area**: Type of area where the applicant resides (e.g., Rural, Semiurban, Urban).  

### Model Outputs  
The model predicts whether the loan will be **approved** or **not approved** using the following labels:  
- `Y`: Loan approved  
- `N`: Loan not approved  

You can test the model by sending a request to the following endpoint:

[https://loan-eligibility-prediction-j102.onrender.com/predict](https://loan-eligibility-prediction-j102.onrender.com/predict)

### Test Input Example  
The following is an example JSON input used to test the model:  

```json  
[{  
    "Gender": "Male",  
    "Married": "Yes",  
    "Dependents": "0",  
    "Education": "Graduate",  
    "Self_Employed": "No",  
    "ApplicantIncome": 5000,  
    "CoapplicantIncome": 2000,  
    "LoanAmount": 150,  
    "Loan_Amount_Term": 360.0,  
    "Credit_History": 0.0,  
    "Property_Area": "Urban"  
} ] 
```  

The model processes this input to predict whether the loan application will be approved or not.  

---

## Limitations  
1. **Bias in Dataset**: The model's performance is directly dependent on the quality and diversity of the training data. If the dataset contains bias (e.g., limited representation of certain demographics), predictions may reflect those biases.  
2. **Feature Dependency**: Features like `Credit_History` and `ApplicantIncome` play a significant role in the predictions, potentially overshadowing less significant features.  
3. **Static Feature Set**: The model does not handle unseen or new categories in features (e.g., a new type of `Property_Area` not present during training).  
4. **Imbalanced Classes**: If the training data is imbalanced (e.g., more approved loans than not approved), it might affect the model's ability to accurately predict minority classes.  
5. **Generalization**: The model may not generalize well to data that significantly deviates from the training dataset.  

---

