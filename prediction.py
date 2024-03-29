from google.cloud import aiplatform # pip install google-cloud-aiplatform

endpoint_id = "<your-endpoint-id>"
project_id = "<your-project-id>"

endpoint = aiplatform.Endpoint(
    endpoint_name=f"projects/{project_id}/locations/us-central1/endpoints/{endpoint_id}"
)

test_mpg = [1.4838871833555929,
 1.8659883497083019,
 2.234620276849616,
 1.0187816540094903,
 -2.530890710602246,
 -1.6046416850441676,
 -0.4651483719733302,
 -0.4952254087173721,
 0.7746763768735953]

response = endpoint.predict([test_mpg])

print('API response: ', response)

print('Predicted MPG: ', response.predictions[0][0])