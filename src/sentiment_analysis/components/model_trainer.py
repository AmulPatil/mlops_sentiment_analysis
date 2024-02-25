import os
import urllib.request as request
from zipfile import ZipFile
import joblib
import time
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from src.sentiment_analysis.entity.config_entity import TrainingConfig
from pathlib import Path
from sklearn.model_selection import train_test_split
import pdb



class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    
    def get_base_model(self):
        self.model = joblib.load(self.config.base_model_path)
    
    
    @staticmethod
    def save_model(path: Path, model: BaseEstimator):
        joblib.dump(model, path)

    def train(self):
        # pdb.set_trace()
        training_path = str(Path(self.config.training_data))
        df_train = pd.read_csv(training_path+'/train.csv')
        
        self.model.fit(df_train['text'], df_train['label'])
        self.save_model(
            path=self.config.updated_base_model_path,
            model=self.model
        )
        

