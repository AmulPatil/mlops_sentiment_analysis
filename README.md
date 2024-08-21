
# Sentiment Analysis with DVC & DagsHub-MLflow Integration

## Overview

This project demonstrates the integration of DVC (Data Version Control) and DagsHub's MLflow for versioning, tracking, and managing machine learning experiments in a sentiment analysis pipeline.

## Project Structure

```plaintext
├── src/sentiment_analysis
│   ├── pipeline/stage_01_data_ingestion.py          # Raw data collection
│   ├── pipeline/stage_02_data_cleaning.py              # Processed data ready for modeling
├── src/sentiment_analysis
│   ├── pipeline/stage_03_prepare_base_model.py             # Script to prepare base model
│   ├── pipeline/stage_04_model_trainer.py                   # Script to train the sentiment analysis model
    ├── pipeline/stage_05_model_evaluation.py                 # Script to evaluate the model
│   
├── artifacts/
│   ├── data_ingestion/raw.gzip             # raw data stored
│   ├── data_cleaning/processed_data.csv             # processed data
│   ├── prepare_base_model/model.pkl             # base model stored
│   ├── training/model.pkl             # Trained sentiment analysis model
├── dvc.yaml                  # DVC pipeline configuration
├── params.yaml               # Parameters for training and evaluation
├── requirements.txt          # Python package dependencies
```

### Track and Visualize Results

- Use DagsHub to visualize MLflow experiments, model metrics, and artifacts.
- Version control your data and models with DVC.

## Conclusion

This project illustrates how to effectively integrate DVC and DagsHub's MLflow for managing data, experiments, and models in a sentiment analysis pipeline.
