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
        features = self._get_features_from_image(attachment_info)
        predictions = []
        for predictor in self.predictors:
            predictions.append(predictor(features))
        return predictions

    def _get_features_from_image(self, attachment_info: dict):
        image = cv2.imread(attachment_info["local_path"])
        output = image.copy()
        output = cv2.resize(output, (128, 128))
        # pre-process the image in the same manner we did earlier
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (200, 200))
        image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        return image

    def _get_tensorflow_prediction(self, features):
        url = 'https://pddstensorflow.azurewebsites.net/predict'
        result = dict()
        result["models"] = self._send_predictor_request(url, features)
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

    def _send_predictor_request(self, url:str, features) -> dict:
        params = {'param0': 'param0', 'param1': 'param1'}
        data = {'params': params, 'features': features.tolist()}

        response = requests.post(url, json=data)
        data = json.loads(response.text)
        return data