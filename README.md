# Customer Churn Prediction
## Week 1: Exploratory Data Analysis

Loads, cleans, analyzes, and visualizes customer churn data to identify 
patterns and relationships between customer features and churn.

### Dataset
- Source: Telco Customer Churn (Kaggle)
- Size: 7,043 customers, 21 features
- Target: Predict customer churn (Yes/No)

### Tools Used
- Python (Pandas, NumPy)
- Matplotlib, Seaborn
- Kaggle Notebooks

## High-Risk Customer Characteristics

### Contract Type
- Month-to-month customers show the highest churn risk.
- Customers with one-year or two-year contracts are more likely to stay.

### Tenure
- Customers with shorter tenure churn more often.
- Long-term customers show greater retention.

### Monthly & Total Charges
- Higher monthly charges are linked with higher churn.
- Total charges are strongly related to tenure.

### Services
- Fiber optic internet customers show a higher churn rate.
- Customers without Online Security / Tech Support are higher-risk.

### Payment Method
- Electronic check users show the highest churn rate.
- Automatic payment methods are associated with better retention.

## Patterns Observed
1. Overall churn rate is approximately 26.5% (1,869 churned out of 7,043 customers).
2. Contract length is strongly related to churn behavior.
3. Tenure and total charges are naturally related.
4. Payment method and internet service type both show clear churn patterns.

## Questions for Next Week
1. Which combination of contract type, tenure, and charges gives the highest churn probability?
2. Do Online Security and Tech Support significantly reduce churn?
3. Can a machine learning model accurately predict which customers will churn?

### Setup
Open the Kaggle notebook or run locally:
```
pip install pandas numpy matplotlib seaborn
```
