# This will be a Python web application that communicates with
# a browser via HTTP (being a web server) and with our deployed model
# via GRPC (being a client)
import sys

from google.cloud import automl_v1beta1
from google.cloud.automl_v1beta1.proto import service_pb2
from google.protobuf.json_format import MessageToJson

# We use additional imports for Flask (web framework)
from flask import Flask
from flask import request

app = Flask(__name__)
# Data won't be taken from command line arguments
project_id = "cnlb-test"
model_id = "TST7502584310504619338"

def get_prediction(content, project_id, model_id):
  prediction_client = automl_v1beta1.PredictionServiceClient()

  name = 'projects/{}/locations/us-central1/models/{}'.format(project_id, model_id)
  payload = {'text_snippet': {'content': content, 'mime_type': 'text/plain' }}
  params = {}
  request = prediction_client.predict(name, payload, params)
  return request

@app.route("/")
def main():
  # The content is taken from the web request
  content = request.args.get('content')
  # Additionaly we convert the response to JSON to display it
  prediction = MessageToJson(get_prediction(content, project_id, model_id))
  return prediction
