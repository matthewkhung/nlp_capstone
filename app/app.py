import os
import streamlit as st
from disaster_tweet_detect import pipelines
import logging


class DisasterTweetDetector:

    def __init__(self):
        if 'pipeline' not in st.session_state:
            #st.session_state.pipeline = pipelines.load_pipeline('./app/disaster_tweet_detect.xgboost.pipeline')
            st.session_state.pipeline = pipelines.load_pipeline_bert('./app/disaster_tweet_detect.albert.keras')
        if 'pred_val' not in st.session_state:
            st.session_state.pred_val = ''

    def predict(self, tweet: str):
        if not st.session_state.pipeline:
            logging.error('No pipeline found')
            return

        st.session_state.pred_val = 'True' if st.session_state.pipeline.predict([tweet])[0] == 1 else 'False'

    def input_tweet_callback(self):
        self.predict(st.session_state.str_tweet)


    def window(self):
        st.markdown("""
        # Disaster Tweet Detector
        
        The Disaster Tweet Detector is a capstone project for the UCSD/Springboard 
        boot camp program. This project aims at identifying Tweets are are 
        discussing disaster events as outlined in the [Kaggle NLP Getting Started](
        https://www.kaggle.com/competitions/nlp-getting-started/)
        project.
        
        Start by pasting in a desired Tweet into the text box below and click 
        predict.
        """)
        str_tweet = st.text_input('Input Tweet', on_change=self.input_tweet_callback, key='str_tweet')
        st.text(f'Is a Tweet about a disaster? {st.session_state.pred_val}')
        st.button('Predict', on_click=self.predict(str_tweet))


if __name__ == "__main__":
    debug = os.getenv('DEBUG', 'false') == 'true'

    if debug:
        app = DisasterTweetDetector()
        app.window()
    else:
        try:
            app = DisasterTweetDetector()
            app.window()
        except Exception as e:
            st.error('Internal error occurred.')
