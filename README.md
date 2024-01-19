

# Vertex AI Custom Training Demo
This repository demonstrates how to train a machine learning model using the MPG dataset and deploy it on Google Cloud's Vertex AI. The process includes environment setup, Docker containerization of training code, and deployment for custom training on Vertex AI.

## Prerequisites

- Google Cloud SDK
- Docker
- Python 3.x

## 1: Set Up Your Environment

Authenticate with Google Cloud:

```
gcloud auth login
gcloud config set project [YOUR_PROJECT_ID]
```

## 2: Containerize Training Code

### Define Environment Variables (Optional)

```
export PROJECT_ID=[YOUR_PROJECT_ID]
export BUCKET_NAME=gs://${PROJECT_ID}-bucket
export IMAGE_URI=gcr.io/$PROJECT_ID/mpg:v1
```

### Create a Cloud Storage Bucket (replace [REGION] with your region):

```
gsutil mb -l [REGION] $BUCKET_NAME
```

## 3: Build and Test the Container Locally

```
docker build ./ -t $IMAGE_URI
docker run $IMAGE_URI
```

## 4: Push the container to Container Registry

```
docker push $IMAGE_URI
```

## 5: Run a Training Job on Vertex AI

- Navigate to Vertex AI in Cloud Console:
- Go to the Models section in Vertex AI. Create a Training Job:
- Set up the training job with the custom container image ($IMAGE_URI).
- Specify the machine type and other settings.
- Use the pre-built prediction container and set the model output path to your GCS bucket.

## 6: Deploy a Model Endpoint

- Once the training job is complete, deploy the trained model to an endpoint in Vertex AI.
- Use the deployed endpoint for making predictions with `prediction.py`.
