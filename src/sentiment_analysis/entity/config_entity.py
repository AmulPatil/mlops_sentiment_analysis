from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path

@dataclass(frozen=True)
class DataCleaningConfig:
    common_dir: Path
    data_cleaned_dir: Path
    data_transformer: Path
    data_non_transformer: Path
    

@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    trained_model_path: Path
    updated_base_model_path: Path
    training_data: Path
    params_c: int
    




@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    base_model_path: Path
    trained_model_path: Path
    updated_base_model_path: Path
    training_data: Path
    params_c: int
    


@dataclass(frozen=True)
class EvaluationConfig:
    path_of_model: Path
    training_data: Path
    all_params: dict
    mlflow_uri: str
    