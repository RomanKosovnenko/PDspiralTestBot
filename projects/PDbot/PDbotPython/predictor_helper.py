import json
import requests
import cv2

from botbuilder.schema import (
    Attachment,
    AttachmentData
)

class PredictorHelper:
    
    def __init__(self):
        self.predictors = [self._get_tensorflow_prediction, self._get_pytorch_prediction, self._get_scikitlearn_prediction]

    def get_prediction(self, attachment_info: dict) -> list:
        """
        Prepare features for predictors and call it
        :param attachment_info
        :return: list of predictions
        """
        features = self._get_features_from_image(attachment_info)
        predictions = []
        for predictor in self.predictors:
            predictions.append(predictor(features))
        return predictions

    def _get_features_from_image(self, attachment_info):
        """
        Prepare features for predictors
        :param attachment_info
        :return: list of predictions
        """
        image = cv2.imread(attachment_info["local_path"])
        output = image.copy()
        output = cv2.resize(output, (128, 128))
        # pre-process the image in the same manner we did earlier
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (200, 200))
        image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        return image

    def _get_tensorflow_prediction(self, features):
        """
        Call tensorflow predictor service to get prediction
        :param features
        :return: list of model predictions
        """
        # url = 'https://pddstensorflow.azurewebsites.net/predict'
        url = 'http://127.0.0.1:5001/predict'
        result = dict()
        result["models"] = self._send_predictor_request(url, features)
        result["predictor"] = "TensorFlow"
        return result
    
    def _get_pytorch_prediction(self, features):
        """
        Call PyTorch predictor service to get prediction
        :param features
        :return: list of model predictions
        """
        # url = 'https://pddspytorch.azurewebsites.net/predict'
        url = 'http://127.0.0.1:5000/predict'
        result = dict()
        result["models"] = self._send_predictor_request(url, features)
        result["predictor"] = "PyTorch"
        return result

    def _get_scikitlearn_prediction(self, features):
        """
        Call Scikit-learn predictor service to get prediction
        :param features
        :return: list of model predictions
        """
        # url = 'https://pddsscikit-learn.azurewebsites.net/predict'
        url = 'http://127.0.0.1:5002/predict'
        result = dict()
        result["models"] = self._send_predictor_request(url, features)
        result["predictor"] = "Scikit-learn"
        return result

    def _send_predictor_request(self, url:str, features) -> dict:
        """
        Send post request and return result
        :param url
        :param: feature
        :return: dict
        """
        data = {'features': features.tolist()}

        response = requests.post(url, json=data)
        data = json.loads(response.text)
        return data