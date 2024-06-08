import mlflow

def log_best_model(rf_model, rf_accuracy, gbt_model, gbt_accuracy):
    best_model, best_accuracy, model_name = (rf_model, rf_accuracy, "iris_rf_model") if rf_accuracy > gbt_accuracy else (gbt_model, gbt_accuracy, "iris_gbt_model")
    with mlflow.start_run():
        model_uri = "runs:/" + mlflow.active_run().info.run_id + "/" + model_name
        mlflow.spark.log_model(best_model, "model")
        client = mlflow.tracking.MlflowClient()
        client.create_registered_model(model_name)
        client.create_model_version(
            name=model_name,
            source=model_uri,
            run_id=mlflow.active_run().info.run_id
        )
