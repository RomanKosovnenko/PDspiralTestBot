from flask import Flask, request

from PDDetectionService import get as get_pd_prediction

app = Flask(__name__)

@app.route("/")
def hello():
    return "Service - Ok"

@app.route("/predict", methods=['POST'])
def get_prediction():
    if 'features' in request.json:
        return get_pd_prediction(request.json['features'])
    else:
        return "No image"

if __name__ == "__main__":
  app.run(debug=False, host='0.0.0.0', port=80)