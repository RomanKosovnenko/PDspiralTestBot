from flask import Flask

from PDDetectionService import get as get_pd_prediction

app = Flask(__name__)

@app.route("/")
def hello():
    return "Service - Ok"

@app.route("/predict")
def get_prediction():
      return get_pd_prediction()