import predict_functions as pf
import PostProcess_FeBio as pp
import datetime

model_path = '/FeBio_Inverse\\Models\\2024_5_6_intermediate_train_22_2024_5_7_modified_train_1el100003pat10000_c36'
Results_Folder = "Mod train and test"
test_data_path = '/FeBio_Inverse\\csv_test\\2024_5_6_intermediate_test_22.csv'
current_date = datetime.datetime.now()
date_prefix = str(current_date.year) + '_' + str(current_date.month)  + '_' + str(current_date.day)
numCompPCA = 2
mod_test_path = '/FeBio_Inverse\\Mod train and test\\2024_5_6_intermediate_test_22_2024_5_7_modified_train.csv'


#mod_test_path, pca1, pcaB = pp.process_features(test_data_path, Results_Folder, date_prefix, numCompPCA)

predict_directory, predicted_path, analysis_path = pf.load_model_to_predict_analysis_plot(model_path, mod_test_path, show=True)