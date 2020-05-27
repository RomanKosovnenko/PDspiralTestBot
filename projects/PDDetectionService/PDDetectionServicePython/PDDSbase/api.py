from flask import Flask
from flask_restful import Resource, Api, reqparse

from PDDetectionService import PDDetection

app = Flask(__name__)
api = Api(app)

api.add_resource(PDDetection, '/predict/')

if __name__ == "__main__":
  app.run(debug=False, host='0.0.0.0')