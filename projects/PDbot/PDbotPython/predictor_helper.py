import json
import requests

from botbuilder.schema import (
    Attachment,
    AttachmentData
)

class PredictorHelper:
    
    def __init__(self):
        self.predictors = [self._get_tensorflow_prediction, self._get_pytorch_prediction, self._get_scikitlearn_prediction]

    def get_prediction(self, attachment: Attachment) -> list:
        predictions = []
        for predictor in self.predictors:
            predictions.append(predictor(attachment))
        return predictions



    def _get_tensorflow_prediction(self, attachment: Attachment):
        url = 'https://pddstensorflow.azurewebsites.net/predict'
        result = self._send_predictor_request(url)
        result["predictor"] = "TensorFlow"
        return result
    
    def _get_pytorch_prediction(self, attachment: Attachment):
        url = 'https://pddspytorch.azurewebsites.net/predict'
        result = self._send_predictor_request(url)
        result["predictor"] = "PyTorch"
        return result

    def _get_scikitlearn_prediction(self, attachment: Attachment):
        url = 'https://pddsscikit-learn.azurewebsites.net/predict'
        result = self._send_predictor_request(url)
        result["predictor"] = "Scikit-learn"
        return result

    def _send_predictor_request(self, url:str, jsonArgs:dict = {}) -> dict:
        r = requests.get(url = url, json = jsonArgs)
        data = r.json()
        return data