import sys

from google.cloud import automl_v1beta1
from google.cloud.automl_v1beta1.proto import service_pb2
# This is required for the GRPC communication. This service uses GRPC and protobuf, not HTTP.


def get_prediction(content, project_id, model_id):
  # We create a client to communicate with the deployed model
  prediction_client = automl_v1beta1.PredictionServiceClient()

  # Determining the model to call and prepare the payload with the text 
  name = 'projects/{}/locations/us-central1/models/{}'.format(project_id, model_id)
  payload = {'text_snippet': {'content': content, 'mime_type': 'text/plain' }}
  params = {}
  request = prediction_client.predict(name, payload, params)
  return request

if __name__ == '__main__':
  content = sys.argv[1]
  project_id = sys.argv[2]
  model_id = sys.argv[3]

  print(get_prediction(content, project_id,  model_id))
