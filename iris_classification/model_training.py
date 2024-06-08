from pyspark.ml.feature import StringIndexer
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import mlflow
import mlflow.spark

def train_and_evaluate_models(iris_df_with_class4):
    # Convert class labels to numeric indices
    indexer = StringIndexer(inputCol="class", outputCol="label")
    indexed_df = indexer.fit(iris_df_with_class4).transform(iris_df_with_class4)
    if "prediction" in indexed_df.columns:
        indexed_df = indexed_df.drop("prediction")
    train_data, test_data = indexed_df.randomSplit([0.8, 0.2], seed=123)

    # Train RandomForest
    rf = RandomForestClassifier(featuresCol="features", labelCol="label")
    rf_model, rf_accuracy = train_model(rf, train_data, test_data)

    # Train GBT
    gbt = GBTClassifier(featuresCol="features", labelCol="label")
    gbt_model, gbt_accuracy = train_model(gbt, train_data, test_data)

    return rf_model, rf_accuracy, gbt_model, gbt_accuracy

def train_model(classifier, train_data, test_data):
    with mlflow.start_run():
        model = classifier.fit(train_data)
        predictions = model.transform(test_data)
        evaluator = MulticlassClassificationEvaluator(labelCol="label", metricName="accuracy")
        accuracy = evaluator.evaluate(predictions)
        mlflow.log_param("Classifier", classifier.__class__.__name__)
        mlflow.log_metric("Accuracy", accuracy)
        mlflow.spark.log_model(model, f"iris_{classifier.__class__.__name__.lower()}_model")
    return model, accuracy
