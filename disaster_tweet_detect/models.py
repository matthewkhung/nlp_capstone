import pandas as pd
import numpy as np
import logging
import pickle
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import xgboost as xgb
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text          # needed for BERT
from official.nlp import optimization   # needed for BERT
import keras
from keras.layers import Layer
import logging

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


# https://stackoverflow.com/questions/59030626/how-to-add-post-processing-into-a-tensorflow-model
class RoundLayer(Layer):
    def __init__(self):
        super(RoundLayer, self).__init__()

    def call(self, inputs):
        return tf.cast(tf.math.round(inputs), tf.int64)


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
                     summary: Summary,
                     retrain: False):

    if retrain:
        model = train_ridge_classifier(X_train, y_train)
    else:
        try:
            # load model from file
            model = pickle.load(open(model_path, "rb"))
        except:
            model = train_ridge_classifier(X_train, y_train)

    # generate prediction
    pred = model.predict(X_test)

    # evaluate and save to summary
    summary.add('Ridge Classifier', y_test, pred)


def train_xgboost_classifier(X_train: np.ndarray, y_train: np.ndarray):
    """Train XGBoost classifier and return model."""
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


def load_xgboost_classifier(model_path: str):
    """Unpickle the model and return it."""
    try:
        # load model from file
        return pickle.load(open(model_path, "rb"))
    except:
        return None


def evaluate_xgboost_classifier(model_path: str,
                                X_train: pd.DataFrame, X_test: pd.DataFrame,
                                y_train: pd.Series, y_test: pd.Series,
                                summary: Summary,
                                retrain: False):
    """Evaluate and add XGBoost classifier results to summary."""
    model = None
    if not retrain:
        model = load_xgboost_classifier(model_path)
    if not model:
        model = train_xgboost_classifier(X_train, y_train)

    # generate prediction
    pred = model.predict(X_test)

    # evaluate and save to summary
    summary.add('XGBoost Classifier', y_test, pred)


def train_bert_classifier(X_train: np.ndarray, y_train: np.ndarray):
    """Train BERT classifier and return model."""
    #https: // www.tensorflow.org / text / tutorials / classify_text_with_bert
    # Architecture
    # text input layer
    # preprocessor: albert_en_preprocess
    # saved model: albert_en_base
    # dropout layer
    # DNN classifier layer

    # split data into train/validation/test
    X_train_bert, X_test_bert, y_train_bert, y_test_bert = train_test_split(X_train, y_train,
                                                                            train_size=0.9, random_state=42)
    X_train_bert, X_val_bert, y_train_bert, y_val_bert = train_test_split(X_train_bert, y_train_bert, train_size=0.8,
                                                                          random_state=42)

    # build classifier model
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string)
    preprocessor = hub.KerasLayer("http://tfhub.dev/tensorflow/albert_en_preprocess/3")
    encoder_inputs = preprocessor(text_input)
    encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/albert_en_base/3", trainable=True)
    outputs = encoder(encoder_inputs)
    net = outputs["pooled_output"]
    net = tf.keras.layers.Dropout(0.1)(net)
    net = tf.keras.layers.Dense(1, activation='sigmoid', name='classifier')(net)
    clf_albert = tf.keras.Model(text_input, net)

    # train model
    loss = tf.losses.BinaryCrossentropy(from_logits=True)
    metrics = tf.metrics.BinaryAccuracy()

    # setup adamw params
    epochs = 5
    batch_size = 64
    steps_per_epoch = len(X_train_bert) / batch_size
    num_train_steps = steps_per_epoch * epochs
    num_warmup_steps = int(0.1 * num_train_steps)

    init_lr = 3e-5
    optimizer = optimization.create_optimizer(init_lr=init_lr,
                                              num_train_steps=num_train_steps,
                                              num_warmup_steps=num_warmup_steps,
                                              optimizer_type='adamw')

    # compile
    clf_albert.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # fit
    history = clf_albert.fit(x=X_train_bert, y=y_train_bert, validation_data=(X_val_bert, y_val_bert),
                             batch_size=batch_size, epochs=epochs)

    # save model
    clf_albert.save('disaster_tweet_detect.albert.keras')
    return clf_albert


def load_bert_classifier(model_path: str):
    """Load keras model."""
    try:
        # load model from file
        clf_bert = tf.keras.models.load_model(model_path,
                                              custom_objects={'KerasLayer': hub.KerasLayer},
                                              compile=False)
        # add rounding layer
        pipe = keras.Sequential()
        pipe.add(clf_bert)
        pipe.add(RoundLayer())
        return pipe
    except Exception as e:
        logging.error(f'Unable to load model: {model_path}')
        logging.error(e)
        return None


def evaluate_bert_classifier(model_path: str,
                                X_train: np.ndarray, X_test: np.ndarray,
                                y_train: np.ndarray, y_test: np.ndarray,
                                summary: Summary,
                                retrain: False):
    """Evaluate and add BERT classifier results to summary."""
    model = None
    if not retrain:
        model = load_bert_classifier(model_path)
    if not model:
        model = train_bert_classifier(X_train, y_train)

    # generate prediction
    pred = model.predict(X_test)

    # evaluate and save to summary
    summary.add('ALBERT Classifier', y_test, pred)
