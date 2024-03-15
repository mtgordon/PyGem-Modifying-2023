import predict_funtions as pf
import PostProcess_FeBio as pp
import PCA_data as pd

import random


def machine_learning_all(epochs, epochs_start, layers, capacity, patience, squared, data=None):
    # TODO: Step 1
    # generate the csv file (all)
    # csv_all = pp.generate_int_csvs(fileTemplate, object_list, logFile, workingInputFileName, first_int_file_flag, csv_filename)
    csv_all = data
    # TODO: STEP 2
    train_data_path, test_data_path = pf.generate_train_test_csvs_files_from(csv_all)
    # load dataset

    # TODO: STEP 3
    # Generate new PCs based on the train file,
    Results_Folder = "Mod train and test"
    mod_train_path, pca1, pcaB = pp.process_features(train_data_path, Results_Folder)

    # Save PCs and save the PCA Model
    # pc_csv_path = pf.generate_PC_csv_file_from(mod_train_path, [5, 6, 7, 8])

    # add noise to test file
    mod_test_path, pca1, pcaB = pp.process_features(train_data_path, Results_Folder)
    noise_test_path = pd.add_noise_to_csv(test_data_path, Results_Folder, pca1, pcaB, noise_scale=0)



    # TEST USE!!!
    # test = pf.generate_train_test_csvs_files_from(mod_test_path)[1]

    # TODO: Step 5

    # model_path, learning_curve_path, predict_directory, predicted_path, analysis_path = pf.machine_learning_save_predict(mod_train_path, mod_test_path, epochs, layers, capacity, patience, epochs_start)
    # Machine learning on train data and predict test data
    # model_path, learning_curve_path = pf.fit_model_save_best_and_curve(mod_train_path, epochs, layers, capacity, patience, epochs_start, squared)
    # #
    # # # Use the train model before to predict the test file above
    # predict_directory, predicted_path, analysis_path = pf.load_model_to_predict_analysis_plot(model_path, mod_test_path, show=True)

    # TODO: Step 6
    # analysis and plot

    # pf.write_predicted_y_to_csv(output_predicted, predicted_y)
    # pf.write_predicted_y_analysed_to_csv(output_analysis, predicted_y, new_y)

    # print("output predicted name: ", output_predicted.title())
    # print("output analysis name: ", output_analysis.title())
    # print("output learning curve: ", output_image.title())
    print("output train data: ", train_data_path)
    print("output test data: ", test_data_path)
    # print("Modified train data: ", mod_train_path)
    print("Modified test data: ", mod_test_path)
    # print("Saved PCs data: ", pc_csv_path)
    print("noise_0 test data: ", noise_test_path)
    # print("Saved best model name: ", model_path)
    # print("Model learning curve: ", learning_curve_path)
    # print("Predicted data folder: ", predict_directory)
    # print("Predicted data file name: ", predicted_path)
    # print("Analysed data file name: ", analysis_path)


old_data = 'combo_features (1).csv'

train_data = 'combo_features (1).csv'
test_data = 'updated_order_features_5_30.csv'
# model_path = 'Models\\real_y2el_50009_c36_x1x2'
epochs = 10000
epochs_start = 10
layers = 3
capacity = 36
patience = 50
squared = False
random.seed(33)
data = "C:\\Users\\yyt08\\PycharmProjects\\PyGem-Modifying-2023\\FeBio Inverse\\intermediate.csv"

machine_learning_all(epochs, epochs_start, layers, capacity, patience, squared, data)
# pf.machine_learning_save_predict(train_data, test_data)
