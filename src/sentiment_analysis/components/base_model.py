import os
import urllib.request as request
# from zipfile import ZipFile
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer

import joblib
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from pathlib import Path
from src.sentiment_analysis.entity.config_entity import PrepareBaseModelConfig
                                                




class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    
    def get_base_model(self):
        self.model  = Pipeline([
            ('vectorizer', CountVectorizer(binary=True)),
            ('classifier', LinearSVC(random_state=10, C=self.config.params))
        ])
        # self.model = LinearSVC(random_state=10,C=self.config.params)

        self.save_model(path=self.config.base_model_path, model=self.model)

    @staticmethod
    def save_model(path: Path, model: BaseEstimator):
        joblib.dump(model, path)
        # model.save(path)
        # model = joblib.load(f'{path}/base_model.pkl')



