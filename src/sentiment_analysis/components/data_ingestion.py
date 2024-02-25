import os
import zipfile
import subprocess,tarfile
from glob import glob
from src.sentiment_analysis import logger
from src.sentiment_analysis.utils.common import get_size
from src.sentiment_analysis.entity.config_entity import DataCleaningConfig


class DataCleaning:
    def __init__(self, config: DataCleaningConfig):
        self.config = config
    
    def data_preprocessing(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        
        list_gz = glob(self.config.local_data_file)
        for single_gz in list_gz:
            with tarfile.open(single_gz, 'r') as tar:
                # Extract all contents to the specified folder
                tar.extractall(path=unzip_path)


