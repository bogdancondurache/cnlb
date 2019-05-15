#!/bin/sh

# This time we use request.json as payload, it contains the image in base64 (open with a lightweight editor, like vim)
curl -X POST -H "Content-Type: application/json" \
  -H "Authorization: Bearer $(gcloud auth application-default print-access-token)" \
  https://automl.googleapis.com/v1beta1/projects/cnlb-test/locations/us-central1/models/IOD896788071893172224:predict \
  -d @request.json
