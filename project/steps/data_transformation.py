import os
import sys
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib
from zenml.steps import step
from typing import Tuple
import seaborn as sns
import matplotlib.pyplot as plt
from logger import logging
import mlflow
from exception import CustomException

@step
def data_transformation(raw_data_path: str) -> Tuple[str, str]:
    try:
        logging.info("Starting data transformation process.")
        data = pd.read_csv(raw_data_path)
        
        # Perform data transformation steps...
        data['Age'] = 2024 - data['Year_Birth']
        data = data[data['Age'] < 80]
        data = data[data['Income'] < 150000]
        data['relationship'] = data['Marital_Status'].replace(
            {'Married': 'in_relationship', 'Together': 'in_relationship', 'Single': 'single',
             'Divorced': 'single', 'YOLO': 'single', 'Absurd': 'single', 'Widow': 'single', 'Alone': 'single'})
        data["Education_Level"] = data["Education"].replace(
            {"Basic": "Undergraduate", "2n Cycle": "Undergraduate", "Graduation": "Graduate",
             "Master": "Postgraduate", "PhD": "Postgraduate"})
        data['children'] = data['Kidhome'] + data['Teenhome']
        data['AcceptedCmp'] = data['AcceptedCmp1'] + data['AcceptedCmp2'] + data['AcceptedCmp3'] + \
                              data['AcceptedCmp4'] + data['AcceptedCmp5'] + data['Response']
        data['num_purchases'] = data['NumWebPurchases'] + data['NumCatalogPurchases'] + data['NumStorePurchases'] + data['NumDealsPurchases']
        data['expenses'] = data['MntWines'] + data['MntFruits'] + data['MntMeatProducts'] + data['MntFishProducts'] + \
                           data['MntSweetProducts'] + data['MntGoldProds']

        data.drop(labels=['Marital_Status', 'ID', 'Year_Birth', 'Dt_Customer', 'Kidhome', 'Teenhome',
                          'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts',
                          'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases',
                          'NumStorePurchases', 'NumWebVisitsMonth', 'AcceptedCmp3', 'AcceptedCmp4',
                          'AcceptedCmp5', 'AcceptedCmp1', 'AcceptedCmp2', 'Z_CostContact', 'Z_Revenue',
                          "Recency", "Complain", 'Education', 'Response', 'AcceptedCmp'], axis=1, inplace=True)

        numerical_cols = ['Income', 'Age', 'children', 'num_purchases', 'expenses']
        ordinal_cols = ['Education_Level']
        nominal_cols = ['relationship']

        num_pipeline = Pipeline(
            steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]
        )

        ordinal_pipeline = Pipeline(
            steps=[
                ('ordinalencoder', OrdinalEncoder(categories=[['Undergraduate', 'Graduate', 'Postgraduate']]))
            ]
        )

        onehot_pipeline = Pipeline(
            steps=[
                ('onehotencoder', OneHotEncoder(categories=[['in_relationship', 'single']]))
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ('num_pipeline', num_pipeline, numerical_cols),
                ('ordinal_pipeline', ordinal_pipeline, ordinal_cols),
                ('onehot_pipeline', onehot_pipeline, nominal_cols)
            ]
        )

        transformed_data = preprocessor.fit_transform(data)
        transformed_data_path = os.path.join("artifacts", "transformed_data.csv")
        transformed_data_df = pd.DataFrame(transformed_data, columns=preprocessor.get_feature_names_out())
        transformed_data_df.to_csv(transformed_data_path, index=False)

        preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
        joblib.dump(preprocessor, preprocessor_path)
        
        # Log artifacts to MLflow
        mlflow.log_artifact(transformed_data_path)
        mlflow.log_artifact(preprocessor_path)

        # Visualize transformed data
        # Age distribution plot after transformation
        sns.histplot(data['Age'], bins=30, kde=True)
        plt.title('Age Distribution After Transformation')
        age_dist_trans_path = os.path.join("artifacts", "age_distribution_after_transformation.png")
        plt.savefig(age_dist_trans_path)
        plt.close()
        mlflow.log_artifact(age_dist_trans_path)

        # Total spending distribution plot after transformation
        sns.histplot(data['expenses'], bins=30, kde=True)
        plt.title('Total Spending Distribution After Transformation')
        total_spending_dist_trans_path = os.path.join("artifacts", "total_spending_distribution_after_transformation.png")
        plt.savefig(total_spending_dist_trans_path)
        plt.close()
        mlflow.log_artifact(total_spending_dist_trans_path)

        logging.info("Data transformation process completed successfully.")
        return transformed_data_path, preprocessor_path
    except Exception as e:
        raise CustomException(e, sys)
