#!/bin/bash

# curl is used for HTTP requests
# This is a POST request to our model deployed on Google Cloud
# The payload is a JSON object with what we want the model to evaluate, in this case
# we evaluate the sentiment for the text "like it"
curl -X POST \
  -H "Authorization: Bearer $(gcloud auth application-default print-access-token)" \
  -H "Content-Type: application/json" \
  https://automl.googleapis.com/v1beta1/projects/cnlb-test/locations/us-central1/models/TST7502584310504619338:predict \
  -d '{
        "payload" : {
          "textSnippet": {
               "content": "like it",
                "mime_type": "text/plain"
           },
        }
      }'
