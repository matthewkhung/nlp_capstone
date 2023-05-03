import keras

from disaster_tweet_detect import models, data
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.wrappers.scikit_learn import KerasClassifier
from keras import Sequential
import pickle
import pandas as pd
import numpy


def train_pipeline_xgboost(X_train: numpy.ndarray, y_train: numpy.ndarray):
    """Train and return a pipeline.

    Parameters
    ----------
    X_train: numpy.ndarray
    y_train: numpy.ndarray
    """

    vectorizer = TfidfVectorizer(strip_accents='ascii',
                                 preprocessor=data.preprocessor,
                                 tokenizer=data.tokenizer,
                                 stop_words=None)
    csr_tfidf = vectorizer.fit_transform(X_train)
    model = models.train_xgboost_classifier(csr_tfidf, y_train)

    pipe = Pipeline(steps=[
        ('vectorizer', vectorizer),
        ('xgboost', model)
    ])

    # save pipeline to file
    pickle.dump(pipe, open("disaster_tweet_detect.xgboost.pipeline", "wb"))
    return pipe


def load_pipeline_bert(pipeline_path: str):
    return models.load_bert_classifier(pipeline_path)


def load_pipeline(pipeline_path: str) -> Pipeline:
    """Unpickle the pipeline and return it."""
    try:
        # load pipeline from file
        return pickle.load(open(pipeline_path, "rb"))
    except:
        return None
