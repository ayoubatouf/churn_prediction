# Telecom Churn Prediction

## Project overview

This repository contains a simple machine learning pipeline designed to predict customer churn in the telecom industry. The project uses classical techniques to process, train, evaluate, and deploy the churn prediction using `AdaBoost classifier` model. The dataset, provided by Bharti Prasad on Kaggle `https://www.kaggle.com/code/bhartiprasad17/customer-churn-prediction/input`, contains 7,043 rows (customers) and 21 feature columns. The success metric we aim to maximize is recall, with the goal of minimizing false negatives (i.e., customers who churn but are incorrectly predicted to stay). This is crucial for telecom businesses as it allows them to proactively engage with at-risk customers.

## Setup and Installation

To set up the environment and install dependencies, follow the steps below:

- Option 1: using `requirements.txt`
```
pip install -r requirements.txt
```
- Option 2: using `setup.py`

Alternatively, if you'd like to install the project as a package : 
```
pip install .
```

## Directory structure
The project has the following directory structure:
```
.
├── config/                     
├── data/                       
├── deployment/                 
├── logs/                       
├── reports/                    
├── results/                   
├── scripts/                    
├── src/                        
├── tests/                     
├── Dockerfile                 
├── docker-compose.yaml         
└── requirements.txt  

```

Key Directories :

- `config/`: code for setting up paths for loading and saving data.
- `data/`: stores raw, processed, and inference data files, including the churn dataset.
- `deployment/`: includes Docker image and MLflow configurations for model deployment.
- `logs/`: logs generated during experimentation and API prediction.
- `reports/` :  EDA plots.
- `scripts/`: scripts for training, inference, and running the entire pipeline.
- `results/`: stores trained models, evaluation results, and scaler objects.
- `src/`:  core source code.
- `tests/`: unit tests for the core modules.


## Project usage
1. Running the Full Pipeline
 
To run the entire churn prediction pipeline, execute the main script located at `/scripts/run_all.py`. This script will:

- Process raw data from `/data/raw/` 
- Train the churn prediction model
- Evaluate the model and save the results to `/results/`
- Perform inference on sample data located in `/data/inference/`

Alternatively, you can run the individual steps of the pipeline:
- Training: run `/scripts/train.py` to train the model.
- Inference: run `/scripts/inference.py` to make predictions on new data.

2. Running the API for prediction 

To serve the model via an API, execute the bash script `/src/serving/fast_api/start_fastapi.sh` to start the FastAPI server. You can also test the API by running the `send_request.sh` script.

3. Running the API as a Docker Container 

To run the API in a Docker container:
- Build the Docker image

`docker build -t churn-prediction .`

- Run the Docker container

`docker run -d -p 8000:8000 churn-prediction`

If you'd like to make frequent changes to the project files while the container is running, you can use Docker Compose to manage the container:

`docker-compose up --build`


