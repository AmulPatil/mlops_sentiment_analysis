
from pathlib import Path
import mlflow
import joblib
import pandas as pd
from sklearn.base import BaseEstimator
from urllib.parse import urlparse
from src.sentiment_analysis.entity.config_entity import EvaluationConfig
from src.sentiment_analysis.utils.common import read_yaml, create_directories,save_json


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    @staticmethod
    def load_model(path: Path) -> BaseEstimator:
        return joblib.load(path)
    

    def evaluation(self):
        training_path = str(Path(self.config.training_data))
        df_test = pd.read_csv(training_path+'/test.csv')
        self.model = self.load_model(self.config.updated_base_model_path)
        self.score = self.model.score(df_test['text'], df_test['label'])
        self.save_score()

    def save_score(self):
        scores = {"accuracy": self.score}
        save_json(path=Path("scores.json"), data=scores)

    
    def log_into_mlflow(self):
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(
                {"accuracy": self.score}
            )
            # Model registry does not work with file store
            if tracking_url_type_store != "file":

                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                mlflow.keras.log_model(self.model, "model", registered_model_name="VGG16Model")
            else:
                mlflow.keras.log_model(self.model, "model")
