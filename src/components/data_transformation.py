import os
import sys
import numpy as np
import pandas as pd

from dataclasses import dataclass
from sklearn.preprocessing import MinMaxScaler

from src.exception import CustomException
from src.logger import logging

@dataclass
class DataTransformationConfig:
    #This is created to define any inputs required for data_transformation.py
    preprocessor_obj_file_path=os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self) -> None:
        self.data_transformation_config = DataTransformationConfig()

    def transform_categorical_columns(df):
        '''
        This function is responsible for data transformation of categorical columns
        '''
        try:
            unwanted_columns = ["dteday"] #Columns which do not serve any purpose in model building
            df = df.drop(columns=unwanted_columns, axis=1)

            cat_features = df.select_dtypes(include = "object").columns
            
            #Transforming categorical variables using dummy variables method
            dummy_dataframes = {}
            for column in cat_features:
                dummy_dataframes[column+"_dummy"] = pd.get_dummies(df[column], drop_first=True)

            for key in dummy_dataframes.keys():
                df = pd.concat([df, dummy_dataframes[key]], axis = 1)
            
            df = df.drop(list(cat_features), axis=1)

            return df
        
        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Reading training and test data completed")

            target_column_name = "cnt"

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Categorical columns transformation initiated")
            input_feature_train_df = (input_feature_train_df)
            input_feature_test_df = DataTransformation.transform_categorical_columns(df=input_feature_test_df)
            logging.info("Categorical columns transformation completed")

            logging.info("Numerical columns transformation initiated")
            num_features = input_feature_train_df.select_dtypes(exclude = "object").columns
            scaler = MinMaxScaler()
            input_feature_train_df[num_features]= scaler.fit_transform(input_feature_train_df[num_features])
            input_feature_test_df[num_features] = scaler.transform(input_feature_test_df[num_features])
            logging.info("Numerical columns transformation completed")

            train_df = pd.concat([input_feature_train_df, target_feature_train_df], axis=1)
            test_df = pd.concat([input_feature_test_df, target_feature_test_df], axis=1)
            logging.info("Transformed dataframes are ready")

            return (train_df, test_df)

        except Exception as e:
            raise CustomException(e,sys)




