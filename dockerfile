# Start from conda base image
FROM python:3.9

# set docker workdir
WORKDIR /usr/src

# copy files related to the disaster_tweet_detect module
COPY disaster_tweet_detect ./disaster_tweet_detect
COPY README.md .
COPY setup.py .
COPY requirements.txt .

# copy necessary files to docker (streamlit)
COPY app ./app

# install pip required modules
RUN pip install --no-cache-dir --upgrade -r requirements.txt  && \
    pip install --no-cache-dir .  && \
    python -m nltk.downloader stopwords  && \
    python -m nltk.downloader punkt

EXPOSE 80

ENTRYPOINT [ "streamlit", "run"]
CMD ["./app/app.py", \
     "--server.port", "80", \
     "--server.enableCORS", "true", \
     "--server.enableXsrfProtection", "true", \
     "--server.enableWebsocketCompression", "true", \
     "--browser.serverAddress", "0.0.0.0", \
     "--browser.serverPort", "80"]