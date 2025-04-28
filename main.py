from src.models.RandomForestClassifier.fire_stats_model_training import preprocess_data
from src.models.DataPreprocessing import extract
from src.models.TabPFN.TabPFN import TabPFN_model

# fire_data = extract()
# buffer_data = train_fire_data(fire_data)
# upload_model_to_s3(
#     pickle_buffer=buffer_data,
#     bucket_name = FIRE_PREDICTION_S3_BUCKET,
#     object_key = FIRE_PREDICTION_OBJECT_KEY
# )

fire_data = extract()
X_train, X_test, y_train, y_test = preprocess_data(fire_data)

random_forest_model(X_train, X_test, y_train, y_test)
fcnn_model(X_train, X_test, y_train, y_test)
TabPFN_model(X_train, X_test, y_train, y_test)




