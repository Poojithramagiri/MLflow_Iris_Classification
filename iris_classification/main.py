from data_preparation import load_and_preprocess_data
from clustering import perform_clustering
from model_training import train_and_evaluate_models
from model_logging import log_best_model

# Load and preprocess data
iris_df_spark = load_and_preprocess_data()

# Perform clustering
iris_df_with_class4 = perform_clustering(iris_df_spark)

# Train and evaluate models
rf_model, rf_accuracy, gbt_model, gbt_accuracy = train_and_evaluate_models(iris_df_with_class4)

# Log and register best model
log_best_model(rf_model, rf_accuracy, gbt_model, gbt_accuracy)
