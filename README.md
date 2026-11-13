# Customer Churn Prediction

Predict which customers of a UK online gift wholesaler will **stop buying in the next 90 days**, so the sales team can contact them before they leave.

**Live app:** https://churn-uci-prediction.streamlit.app/

Final project for the Epsilon AI Machine Learning course.

## The business problem

The wholesaler sells gifts and decorations to thousands of shops and resellers. Customers rarely say they are leaving: **they just stop ordering**. Losing a wholesale customer costs all of their future orders, while a retention call costs a few minutes.

**Question:** on the 1st of every month, using each active customer's purchases from the past year, which customers will make **no purchase in the next 90 days**?

## Dataset

[Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii), UCI Machine Learning Repository: 1,067,371 invoice lines from December 2009 to December 2011, with Invoice, StockCode, Description, Quantity, InvoiceDate, Price, Customer ID and Country.

Chen, D. (2012). *Online Retail II* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CG6D. Licensed under CC BY 4.0.

## Project structure

```
├── 1_Data_Cleaning.ipynb               Clean the raw transactions
├── 2_Feature_Engineering_and_EDA.ipynb Build customer features, the target and the EDA
├── 3_Machine_Learning.ipynb            Preprocessing, models, tuning, testing, saving
├── app.py                              Streamlit app (entry point)
├── app_pages/                          The app's pages
├── features.py                         Feature calculation used by the app (same as notebook 2)
├── utils.py                            App helpers: loading, scoring, explanations
├── assets/                             App illustrations
├── data/
│   ├── online_retail_II.xlsx           Raw dataset
│   ├── cleaned_purchases.csv.gz        Output of notebook 1
│   └── churn_data.csv                  Output of notebook 2
├── models/
│   └── churn_model.pkl                 Saved pipeline (preprocessing + model), output of notebook 3
└── requirements.txt
```

## What was done

### 1. Data cleaning
- Combined the two yearly Excel sheets (1,067,371 lines).
- Removed exact duplicates, lines without a Customer ID, cancellation lines, zero prices, generic Manual entries and non-product codes (postage, fees, discounts, tests).
- Removed **purchases that were cancelled on the same day**, because they never really happened.
- Result: **775,294 clean purchase lines** from 5,851 customers.

### 2. Feature engineering and EDA
- One row = **one customer on the 1st of a month**. Features use only the **previous 365 days**; the target uses only the **next 90 days**.
- **Target:** `Churned = 1` when the customer makes no purchase in the next 90 days.
- 12 features, including recency, frequency and spending (RFM), recent activity, and an engineered **OverdueRatio** (days since the last order divided by the customer's usual gap between orders).
- 10 monthly snapshots (December 2010 to September 2011): **42,832 rows**.
- EDA on the training months only: churn is the majority class (64%), and recency, frequency and spending separate churners most clearly.

### 3. Machine learning
- **Chronological split:** train on December 2010 to May 2011, test on August to September 2011. June and July are left out because their 90-day answers overlap the test period. Cross-validation follows the same rule.
- **Preprocessing pipelines:** median imputation, Standard and Robust scaling, One-Hot Encoding for Country (rare countries grouped).
- **Algorithms compared:** Logistic Regression, SGD, Gaussian Naive Bayes, Decision Tree, Random Forest, XGBoost, and polynomial Logistic Regression (degrees 2-4).
- **Imbalance:** no weighting, class weights and SMOTE compared; no weighting worked best because churn is the majority class.
- **Tuning:** GridSearchCV for Logistic Regression, Random Forest and XGBoost.
- **Feature selection (embedded):** Random Forest importance; removed InvoiceCountLast30Days and AverageOrderValueLast365Days, which repeat other features, keeping 10 (CV F1 82.07 → 81.98). Removing a third feature, including Country, lowered the score further.
- **Final model:** Random Forest (500 trees, max depth 8).

## Results (test: August to September 2011, 8,648 unseen customer snapshots)

| Model | F1 | Precision | Recall | Accuracy |
|---|---:|---:|---:|---:|
| **Random Forest (final, 10 features)** | **76.8%** | **65.1%** | **93.5%** | **69.6%** |
| Baseline: flag every customer as churned | 69.9% | 53.7% | 100.0% | 53.7% |

The model caught **4,342 of 4,642 customers who left**, with 2,328 false alarms. Recall matters most here, because a missed churner costs far more than an unnecessary retention call.

## The app

| Page | What it does |
|---|---|
| **Overview** | The problem, the approach and the results in plain words |
| **At-risk customers** | Every active customer on a chosen date, ranked by churn risk, with a suggested action and a downloadable call list |
| **Customer lookup** | One customer's risk, the reasons behind it and their monthly spending |
| **What-if** | Start from a real example customer, change their history and watch the risk update |

The app calculates the features directly from the cleaned invoices with the same formulas as notebook 2, so users never type features by hand. **1 December 2011**, the latest date in the data, is a true forecast: its next 90 days are not in the dataset.

## How to run

Python 3.11 is recommended.

```bash
pip install -r requirements.txt
```

**Notebooks:** run them in order (1 → 2 → 3) from the project folder. Each notebook saves the file the next one reads. Notebook 3 takes about 8 minutes because of the grid searches.

**App:**

```bash
streamlit run app.py
```

## Limitations

- Many false alarms in the pre-Christmas test months, when fewer customers leave than in the training months.
- The data contains purchases only; complaints, deliveries or sales contacts could explain some churn the model misses.
- The deployed model was refitted on all ten months, so app scores for those months are demonstrations; 1 December 2011 is the true forecast.
- A prediction is not a cause: contacting a customer is not proven to change the outcome.
