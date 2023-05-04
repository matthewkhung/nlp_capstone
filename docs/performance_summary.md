# Performance Summary

## Results

Based on the test results below, ALBERT has the highest accuracy and AUROC.

|               Model | Accuracy | AUROC |
|--------------------:|----------|-------|
|       Majority Vote | 0.574    | 0.500 |
| Logistic Regression | 0.784    | 0.761 |
|    Ridge Classifier | 0.775    | 0.755 |
|  XGBoost Classifier | 0.738    | 0.710 |
|              ALBERT | 0.815    | 0.805 |

## Sample Preparation

Samples were prepared by splitting the training data set into train and test sets. In the case of DNNs, the train set
was further split into train and validation sets as cross-validation was done manually. The train and validation sets
were used for training and cross-validation purposes while the test set was used to evaluate the model and was only used
once per model.

## Models

The following models were evaluated:

**Majority Vote (baseline)**: Model always returns the output of the largest target population. This model will be used
as a baseline for the others.

**Logistic Regression Classifier**: Basic logistic regression classifier fitted on the TFIDF vector and the target.

**Ridge Classifier**: Ridge regression models help improve prediction scores on data that suffers from
multicollinearity.

**XGBoost Classifier**: Based on hyperparameter tuning, the model has a max depth of 2 and 180 estimators.

**ALBERT**: ALBERT (A Lite BERT) is a lite version of the deep learning model BERT and offers high performance for NLP
applications.