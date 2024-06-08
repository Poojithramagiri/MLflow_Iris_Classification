import pandas as pd
from sklearn.datasets import load_iris
from pyspark.sql import SparkSession

def load_and_preprocess_data():
    # Load Iris dataset
    iris_sklearn = load_iris()

    # Convert to DataFrame
    iris_df_pd = pd.DataFrame(data=iris_sklearn.data, columns=iris_sklearn.feature_names)
    iris_df_pd['target'] = iris_sklearn.target

    # Convert target integers to class labels
    iris_df_pd['target'] = iris_df_pd['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

    # Initialize SparkSession
    spark = SparkSession.builder \
        .appName("IrisDataset") \
        .getOrCreate()

    # Convert pandas DataFrame to PySpark DataFrame
    iris_df_spark = spark.createDataFrame(iris_df_pd)
    return iris_df_spark
