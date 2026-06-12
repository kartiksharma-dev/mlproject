import sys 
from dataclasses import dataclass
import  numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer #(To Create a Pipeline)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomEXception
from src.logger import logging
import os
from src.utlis import save_object

@dataclass
class DataTransformationconfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransfromation:
    def __init__(self):
        self.data_transformation_config = DataTransformationconfig()

    def get_data_transformation_object(self):

        '''
        This function is resopnisble for data transformation based on 
        different types of data

        '''
        try:
            numerical_columns=["writing score","reading score"]
            categroical_columns=[
               'gender', 
               'race/ethnicity', 
               'parental level of education', 
               'lunch',
                'test preparation course'
            ]

            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )

            cat_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                ]

            )
            logging.info("Numerical columns Standard Scaling completed")
            logging.info("categroical columns encoding completed")
            
            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("cat_pipeline",cat_pipeline,categroical_columns)
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomEXception(e,sys)
            
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
    
            logging.info("Read Train and Test data completed")
            logging.info("obtaining preprocessing object")
    
            preprocessing_obj=self.get_data_transformation_object()
            target_columns_name="math score"
            numerical_columns=["writing score","reading score"]
            
            input_feature_train_df=train_df.drop(columns=[target_columns_name],axis=1)
            target_feature_train_df=train_df[target_columns_name]
            input_feature_test_df=test_df.drop(columns=[target_columns_name],axis=1)
            target_feature_test_df=test_df[target_columns_name]
    
            logging.info(
                f"Applying Preprocessing Object On training Dataframe And Tesing Dataframe."
            )
    
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.fit_transform(input_feature_test_df)
    
            train_arr=np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)
            ]
    
            test_arr=np.c_[
                input_feature_test_arr,np.array(target_feature_test_df)
            ]
    
            logging.info(f"Saved Preprocessing Object.")
    
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
    
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
    
            )
        except Exception as e:
            raise CustomEXception(e,sys)
            