# Credit Scoring Model

## Overview
This project is an AI-powered Credit Scoring System that predicts the creditworthiness of loan applicants using Machine Learning techniques. The system analyzes demographic, employment, income, and housing information to estimate the probability of default and generate a credit score.

## Features
- Data Cleaning and Preprocessing
- Feature Engineering
- Logistic Regression Model
- Decision Tree Model
- Random Forest Model
- ROC-AUC, Precision, Recall, F1-Score Evaluation
- Interactive Streamlit Dashboard
- Credit Score Estimation (300–850 Scale)
- Loan Approval/Rejection Decision Support

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Seaborn
- Streamlit

## Machine Learning Workflow
1. Data Collection
2. Data Preprocessing
3. Feature Engineering
4. Model Training
5. Model Evaluation
6. Credit Risk Prediction
7. Dashboard Deployment

## Model Performance
- Logistic Regression
- Decision Tree
- Random Forest (Best Model)

Random Forest achieved the highest ROC-AUC score and was selected as the final model for deployment.
## How the System Works

The Credit Scoring Model follows a complete Machine Learning workflow to evaluate an applicant's creditworthiness and predict whether a loan should be approved or rejected.

### Working Flow

```text
Applicant Information
        ↓
Data Collection
        ↓
Feature Engineering
        ↓
Data Preprocessing
        ↓
Machine Learning Model
(Random Forest)
        ↓
Probability of Default Calculation
        ↓
Credit Score Generation
        ↓
Loan Approval / Rejection Decision
```

### Step 1: Applicant Information

The user enters personal and financial details through the Streamlit dashboard, including:

* Age
* Gender
* Annual Income
* Employment Status
* Years of Employment
* Family Size
* Housing Type
* Property Ownership
* Vehicle Ownership
* Occupation

### Step 2: Feature Engineering

The system transforms raw data into meaningful features that improve prediction accuracy.

Examples:

* Convert age from birth date information.
* Calculate years of employment.
* Apply logarithmic transformation to income.
* Create employment status indicators.

### Step 3: Data Preprocessing

Before prediction, the data is prepared using:

* Missing value handling
* Numerical feature scaling using StandardScaler
* Categorical feature encoding using One-Hot Encoding

This ensures that all features are in a format suitable for Machine Learning models.

### Step 4: Credit Risk Prediction

The processed applicant data is passed to the trained Random Forest model.

The model was trained using historical credit records and application data, where customers were classified as:

* Good Credit Risk (0)
* Bad Credit Risk (1)

The model learns patterns associated with loan repayment behavior and credit defaults.

### Step 5: Probability of Default Estimation

Instead of directly predicting approval or rejection, the model first calculates:

* Probability of Default (Risk of Non-Payment)
* Probability of Approval (Likelihood of Successful Repayment)

Example:

```text
Probability of Default = 8%
Probability of Approval = 92%
```

### Step 6: Credit Score Calculation

The default probability is converted into an estimated credit score on a scale of:

```text
300 → Highest Risk
850 → Lowest Risk
```

Example:

```text
Default Risk = 5%
Credit Score = 800+
```

Higher scores indicate stronger creditworthiness and lower financial risk.

### Step 7: Loan Approval Decision

The system uses a predefined risk threshold.

```text
Default Probability < 15%
          ↓
       APPROVED

Default Probability ≥ 15%
          ↓
       REJECTED
```

Applicants with low predicted risk are approved, while applicants with higher risk are rejected to minimize potential financial losses.

## Machine Learning Models Used

The project evaluates multiple classification algorithms:

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier

The models are compared using:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC Score

The Random Forest model achieved the best performance and was selected as the final production model.

## Loan Approval Prediction Logic

The system predicts loan approval by analyzing relationships between applicant characteristics and historical repayment behavior.

Factors that typically increase approval chances:

* Higher income
* Stable employment history
* Property ownership
* Longer work experience
* Smaller financial dependency ratio

Factors that may increase rejection risk:

* Low income
* Short employment history
* Large family responsibilities
* Lack of assets
* Historical delinquency patterns

Using these patterns, the model estimates the likelihood of future repayment and automatically generates an approval decision.


## Live Demo
https://credit-score-model-ytg3.onrender.com/

## Author
Arasu Kumar S
