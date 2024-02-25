import os
import zipfile,re
import pandas as pd
import pdb
import subprocess,tarfile
from src.sentiment_analysis import logger
from src.sentiment_analysis.utils.common import get_size
from src.sentiment_analysis.entity.config_entity import DataCleaningConfig
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
english_stop_words = stopwords.words('english')

class DataCleaning:
    def __init__(self, config: DataCleaningConfig):
        self.config = config

    
    def read_and_tag_data(self,folder_path: str) -> pd.DataFrame:
        # pdb.set_trace()
        data = {'file_name': [], 'text': [], 'sentiment': []}
        # print(folder_path)
        for sentiment, label in [('pos', 1), ('neg', 0)]:
            folder = os.path.join(folder_path, sentiment)
            for file_name in os.listdir(folder):
                file_path = os.path.join(folder, file_name)
                with open(file_path, 'r', encoding='utf-8') as file:
                    # print(file_path)
                    text = file.read()
                    data['file_name'].append(file_name)
                    data['text'].append(text)
                    data['sentiment'].append(label)

        df = pd.DataFrame(data)
        return df
    
    def preprocess_reviews(self,reviews:list) -> list:
        REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])|(\d+)")
        REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
        NO_SPACE = ""
        SPACE = " "

        reviews = [REPLACE_NO_SPACE.sub(NO_SPACE, line.lower()) for line in reviews]
        reviews = [REPLACE_WITH_SPACE.sub(SPACE, line) for line in reviews]
        
        return reviews

    def remove_stop_words(self,corpus:list) -> list:
        removed_stop_words = []
        for review in corpus:
            removed_stop_words.append(
                ' '.join([word for word in review.split() 
                        if word not in english_stop_words])
            )
        return removed_stop_words

    def get_stemmed_text(self,corpus:list) ->list:
        stemmer = PorterStemmer()
        return [' '.join([stemmer.stem(word) for word in review.split()]) for review in corpus]


    def get_lemmatized_text(self,corpus:list) ->list:
        lemmatizer = WordNetLemmatizer()
        return [' '.join([lemmatizer.lemmatize(word) for word in review.split()]) for review in corpus]

    def data_preprocessing_model_high_complexity(self,df:pd.DataFrame) -> pd.DataFrame:
        try:
            reviews_train_clean = self.preprocess_reviews(list(df['text']))
            train_target = list(df['sentiment'])
            dataset = pd.DataFrame({'text':reviews_train_clean,'label':train_target})
            return dataset
        except Exception as e:
            raise e
        
    def data_preprocessing_model_simple_complexity(self,df:pd.DataFrame) -> pd.DataFrame:
        try:
            reviews_train_clean = self.get_lemmatized_text(self.get_stemmed_text(self.remove_stop_words(self.preprocess_reviews(list(df['text'])))))
            train_target = list(df['sentiment'])
            dataset = pd.DataFrame({'text':reviews_train_clean,'label':train_target})
            return dataset
        except Exception as e:
            raise e
    
    def data_preprocessing_first_stage(self):
        # pdb.set_trace()
        '''
        Fetch data from the url
        '''
        try: 
            common_dir_path = self.config.common_dir
            data_cleaned_dir = self.config.data_cleaned_dir
            data_transformer = self.config.data_transformer
            data_non_transformer = self.config.data_non_transformer
            main_folders = [entry.name for entry in os.scandir(common_dir_path) if entry.is_dir()]
            sub_folders = [entry.name for entry in os.scandir(common_dir_path+'/'+main_folders[0]) if entry.is_dir()]
            for subs_str in sub_folders:
                df_inter = self.read_and_tag_data(f'{common_dir_path}/{main_folders[0]}/{subs_str}')#.to_csv(f'{data_cleaned_dir}/{subs_str}.csv',index=False)
                self.data_preprocessing_model_simple_complexity(df_inter).to_csv(f'{data_non_transformer}/{subs_str}.csv',index=False)
                self.data_preprocessing_model_high_complexity(df_inter).to_csv(f'{data_transformer}/{subs_str}.csv',index=False)

                logger.info(f"supervised_dataset_created :{f'{data_non_transformer}/{subs_str}.csv'}")

        except Exception as e:
            raise e
        
    

