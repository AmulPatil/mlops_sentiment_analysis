from src.sentiment_analysis.config.configuration import ConfigurationManager
from src.sentiment_analysis.components.data_cleaning import DataCleaning
from src.sentiment_analysis import logger



STAGE_NAME = "Data cleaning stage"

class DataCleaningPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_cleaning_config = config.get_data_cleaning_config()
        data_cleaning = DataCleaning(config=data_cleaning_config)
        data_cleaning.data_preprocessing_first_stage()




if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataCleaningPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e

