import pandas as pd
import numpy as np
import logging
import pickle
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import GridSearchCV
import xgboost as xgb


# helper function to summarize statistics
class Summary:
    def __init__(self):
        self.summary = pd.DataFrame(columns=['model', 'acc', 'roc_auc'])

    def add(self, model_name: str, y_true: pd.Series, y_pred: pd.Series):
        # calculate metrics
        acc = accuracy_score(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_pred)

        # store metrics
        self.summary = pd.concat([self.summary,
                                 pd.DataFrame.from_records(
                                     [{'model':model_name,
                                       'acc':acc,
                                       'roc_auc':roc_auc}])])
        logging.info(f'{model_name} accuracy: {acc:0.6}')
        logging.info(f'{model_name} roc_auc: {roc_auc:0.6}')

    def __repr__(self):
        return str(self.summary)


def evaluate_majority_vote_classifier(X_train: pd.DataFrame, X_test: pd.DataFrame,
                                      y_train: pd.Series, y_test: pd.Series,
                                      summary: Summary):
    # build a majority vote classifier
    majority_vote = round(y_train.mean())
    pred = [majority_vote]*X_test.shape[0]

    # evaluate and save to summary
    summary.add('Majority Vote Classifier', y_test, pred)


def evaluate_logistic_regression_classifier(X_train: pd.DataFrame, X_test: pd.DataFrame,
                                            y_train: pd.Series, y_test: pd.Series,
                                            summary: Summary):
    # logistic regression classifier
    clf_lr = LogisticRegression()
    clf_lr.fit(X_train, y_train)
    pred = clf_lr.predict(X_test)

    # evaluate and save to summary
    summary.add('Logistic Regression Classifier', y_test, pred)


def train_ridge_classifier(X_train: pd.DataFrame, y_train: pd.Series):
    # use grid search to fine-tune params
    params = {
        'alpha': np.arange(0.6,0.7,0.01)
    }

    # build classifier
    clf_ridge = RidgeClassifier()

    # train
    model = GridSearchCV(clf_ridge, params, n_jobs=4, cv=5)
    model.fit(X_train, y_train)
    logging.info('Ridge model fitted')

    # save model to file
    pickle.dump(model, open("disaster_tweet_detect.ridge.model", "wb"))
    return model


def evaluate_ridge_classifier(model_path: str,
                     X_train: pd.DataFrame, X_test: pd.DataFrame,
                     y_train: pd.Series, y_test: pd.Series,
                     summary: Summary):
    try:
        # load model from file
        model = pickle.load(open(model_path, "rb"))
    except:
        model = train_ridge_classifier(X_train, y_train)

    # generate prediction
    pred = model.predict(X_test)

    # evaluate and save to summary
    summary.add('Ridge Classifier', y_test, pred)


def train_xgboost_classifier(X_train: pd.DataFrame, y_train: pd.Series):
    # XGBoost classifier
    # xgboost manual: https://xgboost.readthedocs.io/en/stable/parameter.html
    # tuning guide: https://www.kaggle.com/code/prashant111/a-guide-on-xgboost-hyperparameters-tuning/notebook
    # https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/

    # use grid search to fine-tune each objective
    search_param = {
        # 'max_depth': randint(2,10),                   # search objective 1
        # 'min_child_weight': randint(0, 10),           # search objective 1
        # 'gamma': [0,.1,.2,.3,.4],                     # search objective 2
        # 'subsample': np.arange(0.45,0.65,0.05),       # search objective 3
        # 'colsample_bytree': np.arange(0.85,1,0.05),   # search objective 3
        # 'reg_alpha': [1e-5,1e-4,1e-3],                # search objective 4
    }

    # these are the final params from search
    model_param = {
        'max_depth': 2,             # results from search objective 1
        'min_child_weight': 1,      # results from search objective 1
        'gamma': 0.2,               # results from search objective 2
        'subsample': 0.6,           # results from search objective 3
        'colsample_bytree': 0.95,   # results from search objective 3
        'reg_alpha': 1e-3,          # results from search objective 4
        # 'verbosity': 3,             # info
        'n_estimators': 180,
        'eta': 0.3,                 # default
    }

    # build the classifier
    clf_xgb = xgb.XGBClassifier(objective='binary:logistic', **model_param)

    # train
    model = GridSearchCV(clf_xgb, search_param, scoring='roc_auc',
                         n_jobs=4, cv=5, verbose=3)
    model.fit(X_train, y_train)
    logging.info('XGBoost model fitted')

    # save model to file
    pickle.dump(model, open("disaster_tweet_detect.xgboost.model", "wb"))
    return model


def evaluate_xgboost_classifier(model_path: str,
                                X_train: pd.DataFrame, X_test: pd.DataFrame,
                                y_train: pd.Series, y_test: pd.Series,
                                summary: Summary):
    try:
        # load model from file
        model = pickle.load(open(model_path, "rb"))
    except:
        model = train_xgboost_classifier(X_train, y_train)

    # generate prediction
    pred = model.predict(X_test)

    # evaluate and save to summary
    summary.add('XGBoost Classifier', y_test, pred)

