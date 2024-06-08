from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql.functions import when

def perform_clustering(iris_df_spark):
    # Assemble features
    feature_columns = iris_df_spark.columns[:-1]
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    kmeans = KMeans(k=2, seed=1)  # We cluster into 2 because we're finding similar classes
    pipeline = Pipeline(stages=[assembler, kmeans])
    model = pipeline.fit(iris_df_spark)

    # Assign class label based on cluster
    iris_df_with_cluster = model.transform(iris_df_spark)
    iris_df_with_class4 = iris_df_with_cluster.withColumn("class", when(iris_df_with_cluster["prediction"] == 0, "Class3").otherwise("Class4"))
    return iris_df_with_class4
