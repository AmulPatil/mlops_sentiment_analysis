
# Sentiment Analysis with DVC & DagsHub-MLflow Integration

## Overview

This project demonstrates the integration of DVC (Data Version Control) and DagsHub's MLflow for versioning, tracking, and managing machine learning experiments in a sentiment analysis pipeline.

## Project Structure

```plaintext
├── data/
│   ├── raw/                  # Raw data files
│   ├── processed/            # Processed data ready for modeling
├── src/
│   ├── train.py              # Script to train the sentiment analysis model
│   ├── evaluate.py           # Script to evaluate the model
│   ├── preprocess.py         # Data preprocessing script
├── models/
│   ├── model.pkl             # Trained sentiment analysis model
├── dvc.yaml                  # DVC pipeline configuration
├── params.yaml               # Parameters for training and evaluation
├── requirements.txt          # Python package dependencies
├── README.md                 # Project documentation
└── .dvc/                     # DVC metadata
```

## Setup Instructions

### Step 1: Install Dependencies

Ensure Python 3.7+ is installed, then install the required packages:

```bash
pip install -r requirements.txt
```

### Step 2: DVC Setup

1. Initialize DVC in the project:

    ```bash
    dvc init
    ```

2. Add the data files to DVC:

    ```bash
    dvc add data/raw/
    ```

3. Track the `data/` directory with Git:

    ```bash
    git add data/.gitignore data/raw.dvc
    git commit -m "Add raw data to DVC"
    ```

### Step 3: DagsHub-MLflow Integration

1. Connect the project to DagsHub for MLflow tracking:

    - Link your repository with DagsHub.
    - Configure MLflow tracking URI to point to DagsHub:

    ```bash
    export MLFLOW_TRACKING_URI=https://dagshub.com/your-username/your-repo.mlflow
    ```

2. Run the training script with MLflow logging:

    ```bash
    python src/train.py
    ```

### Step 4: Run the DVC Pipeline

1. Define the DVC pipeline in `dvc.yaml` and link stages:

    ```bash
    dvc run -n preprocess -d src/preprocess.py -o data/processed/ python src/preprocess.py
    dvc run -n train -d src/train.py -d data/processed/ -o models/model.pkl python src/train.py
    ```

2. Execute the pipeline:

    ```bash
    dvc repro
    ```

### Step 5: Track and Visualize Results

- Use DagsHub to visualize MLflow experiments, model metrics, and artifacts.
- Version control your data and models with DVC.

## Conclusion

This project illustrates how to effectively integrate DVC and DagsHub's MLflow for managing data, experiments, and models in a sentiment analysis pipeline.
```
