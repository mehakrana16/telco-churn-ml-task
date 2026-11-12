# Model Evaluation & Error Analysis (Telco Customer Churn)

**Model:** XGBoost (tuned), version 20261107_162457
**Test metrics:** F1=0.556, ROC-AUC=0.840, Average Precision=0.647
**Baselines (5-fold CV F1):** random guess=0.269, LogReg=0.629, RandomForest=0.544, XGBoost tuned=0.593

The tuned model beats the random-guess baseline by a wide margin, but recall for the churn class is low: the model misses about half of the real churners.

Note: Logistic Regression (class_weight='balanced') has a higher CV F1 (0.629) than the tuned XGBoost (0.593) because it trades precision for recall. XGBoost was tuned for F1 but without class weighting, so adding `scale_pos_weight` to the search is the next step.

## Overall error picture
- Test set has 1405 customers, model made 294 errors: **188 missed churn (FN)** and **106 false alarms (FP)**.
- Most errors are missed churners (64%), so the model is conservative: it tends to say "No churn".
- Reviewed the 20 most confident errors (`misclassified_top20.csv`): 20 of 20 are missed churners, predicted "No" with confidence between 0.91 and 0.99.

## Failure Mode 1: Confident misses on customers who look loyal
- **What we saw:** In the top-20 confident errors 14 of the 20 are on a one/two-year contract (7 two-year, 7 one-year, 6 month-to-month); by internet service, 10 are DSL, 8 have no internet and 2 are fiber optic. Most of them have a long contract, long tenure, or no internet service (low bill ~ $20), yet they churned. This pattern covers only the most confident errors: overall just 42 of 188 missed churners (22%) are not on a month-to-month contract, so most missed churners belong to Failure Mode 2.
- **Hypothesis:** The model learned that long contract + high tenure + auto-pay means "stays". These customers fit that pattern, so it is very confident, but their real reason for leaving (competitor offer, moving, price, service issue) is not captured in the available columns. The "no internet" segment also has a low base churn rate, so the model treats the whole segment as safe and cannot isolate its few churners.
- **Data-level fix:** Add features that capture the trigger for leaving: support calls / complaints, drop in usage, days to contract renewal, failed payments, recent plan or price change. Collect more labelled churn examples from the low-churn segments so the model sees them.

## Failure Mode 2: Month-to-month and short-tenure customers are hard to separate
- **What we saw:** 252 of 294 errors (86%) are month-to-month customers. Churn rate by contract is 42.6% (month-to-month), 11.3% (one year) and 2.8% (two year). Among month-to-month customers in the test set the model reaches recall=0.56 and precision=0.63. Misclassified customers have a mean tenure of 20.6 months vs 32.6 for the whole test set, so errors concentrate on newer customers (partly because month-to-month customers naturally have shorter tenure).
- **Hypothesis:** Month-to-month is a high-risk but mixed group: a large share churn and a large share stay. Inside it, the available features (contract, tenure, charges, services) look similar for both outcomes, so the model cannot draw a clean boundary and makes both FP and FN errors. New customers also have little history, so their behaviour is noisy.
- **Data-level fix:** Add behavioural history features (usage trend in first months, onboarding/promotion end date, number of service changes), tenure buckets, `MonthlyCharges/tenure`, and services-count. Gather more data for early-tenure customers.

## Class imbalance and threshold
- Churn is about 27% of the data and the default 0.5 threshold favours the majority class, which partly explains recall of only 0.49 (the tuned XGBoost was also trained without class weights).
- Fix: class weights / `scale_pos_weight`, resampling (e.g. SMOTE) only inside the training folds, and a lower decision threshold chosen on validation or CV data, not on the test set.

## Proposed data-level fixes (summary)
1. Add behavioural and event features (support calls, usage trend, renewal date, payment failures).
2. Add engineered features: tenure buckets, services count, charges per month of tenure.
3. Collect more labelled examples for new customers and for low-churn segments (long contract, no internet).
4. Handle imbalance in training only (class weights / SMOTE inside CV) and tune the threshold on validation data.
5. Re-check labels and features for the confident errors for possible data quality issues.
