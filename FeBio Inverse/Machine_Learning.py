import predict_funtions as pf
import PostProcess_FeBio as pp
import PCA_data as pd

import random


# pf.machine_learning_save_predict(train_data, test_data)


def machine_learning_all(epochs, epochs_start, layers, capacity, patience, squared, data=None):
    # TODO: Step 1
    # generate the csv file (all)
    # csv_all = pp.generate_int_csvs(fileTemplate, object_list, logFile, workingInputFileName, first_int_file_flag, csv_filename)
    csv_all = data
    # TODO: STEP 2
    train_data_path, test_data_path = pf.generate_train_test_csvs_files_from(csv_all)
    # load dataset

    # TODO: STEP 3
    # Generate new PCs based on the train file
    Results_Folder = '2023_6_9_auto'
    # mod_train_path = pp.process_features(train_data_path, Results_Folder, "june, 12")
    #
    # pc_csv_path = pf.generate_PC_csv_file_from(mod_train_path, [5, 6, 7, 8])
    #
    # model_path = pf.fit_model_save_best_and_curve(train_data_path, epochs, layers, capacity, patience, epochs_start, squared)

    # Save PCs and save the PCA Model

    # TODO: Step 4
    # Function to add noice to test file and other features
    mod_test_path = pd.add_noise_to_csv(test_data_path, Results_Folder)

    # TODO: Step 5
    # Use the train model before to predict the test file above
    # pf.load_model_to_predict_analysis_plot(...)

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
    # print("PCs data: ", pc_csv_path)
    # print("Saved best model name: ", model_path)




old_data = 'combo_features (1).csv'

train_data = 'combo_features (1).csv'
test_data = 'updated_order_features_5_30.csv'
# model_path = 'Models\\real_y2el_50009_c36_x1x2'
epochs = 8000
epochs_start = 1000
layers = 9
capacity = 36
patience = 500
squared = False
random.seed(33)
data = "D:\\Gordon\\Automate FEB Runs\\2023_6_9_auto\\2023_6_9_intermediate.csv"

machine_learning_all(epochs, epochs_start, layers, capacity, patience, squared, data)



