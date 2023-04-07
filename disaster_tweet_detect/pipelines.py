from disaster_tweet_detect import models, data
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import pandas as pd
import numpy


def train_pipeline(X_train: numpy.ndarray, y_train: numpy.ndarray):
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
    pickle.dump(pipe, open("disaster_tweet_detect.pipeline", "wb"))
    return pipe


def load_pipeline(pipeline_path: str) -> Pipeline:
    """Unpickle the pipeline and return it."""
    try:
        # load pipeline from file
        return pickle.load(open(pipeline_path, "rb"))
    except:
        return None
