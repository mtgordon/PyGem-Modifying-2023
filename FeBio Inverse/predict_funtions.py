# Author: Yutian Yang
# Created: 5/23/2023
# Description: This is a file of all the machine learning functions
# Version: 1.0
# Email: yyt542985333@gmail.com

import os
from statistics import stdev
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dense, BatchNormalization, Activation, Dropout
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from numpy import mean
from numpy import std
from pandas import read_csv
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split
# from real.PoltFunctions import plot_2_outputs, get_data_corresponding
import tensorflow as tf


# https://machinelearningmastery.com/deep-learning-models-for-multi-output-regression/
def get_available_gpus():
    """
    Get a list of available GPUs on the system.

    Returns:
        List[str]: A list of GPU names.

    Description:
        This function checks for available GPUs on the system using TensorFlow's list_physical_devices() function.
        It returns a list of GPU names if GPUs are available, otherwise it returns an empty list.

    Example:
        >>> available_gpus = get_available_gpus()
        >>> print(available_gpus)
        ['GPU:0', 'GPU:1']

    Notes:
        - This function relies on TensorFlow's GPU support and may not work if TensorFlow is not properly installed
          or configured with GPU support.
        - The returned GPU names can be used to specify the target GPU device for TensorFlow operations.
    """
    local_devices = tf.config.list_physical_devices('GPU')
    gpu_names = [device.name for device in local_devices]

    if gpu_names:
        print("Available GPUs:")
        for gpu_name in gpu_names:
            print(gpu_name)
    else:
        print("No GPUs available.")
    return gpu_names

def set_up_gpu():
    """
    Set up GPU memory allocation for TensorFlow.

    Description:
        This function sets up GPU memory allocation for TensorFlow to allow dynamic memory growth.
        It checks for available GPUs using TensorFlow's list_physical_devices() function and
        enables memory growth for the first GPU if any GPUs are available.

    Notes:
        - This function should be called before using TensorFlow with GPU to ensure proper memory allocation.
        - Enabling memory growth allows TensorFlow to allocate memory on the GPU as needed, which can prevent
          GPU memory errors when working with large models or datasets.
        - If no GPUs are available or an error occurs during the setup, an appropriate message will be printed.

    Example:
        >>> set_up_gpu()

    """
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Allow TensorFlow to allocate GPU memory dynamically
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError as e:
            print(e)


def get_file_name(file_path):
    """
    Get the file name without the extension from a given file path.

    Parameters:
        file_path (str): The path of the file.

    Returns:
        str: The file name without the extension.

    Example:
        >>> file_path = "/path/to/file.txt"
        >>> file_name = get_file_name(file_path)
        >>> print(file_name)
        "file"

    """
    file_name = os.path.basename(file_path)  # Get the base file name
    file_name_without_extension = os.path.splitext(file_name)[0]  # Remove the extension
    return file_name_without_extension


# def normalize(column):
#     """
#     Normalize a column of data by subtracting the mean and dividing by the standard deviation.
#
#     Parameters:
#         column (array-like): The column of data to be normalized.
#
#     Returns:
#         array-like: The normalized column of data.
#
#     Notes:
#         - The normalization is performed by subtracting the mean of the column from each element
#           and dividing the result by the standard deviation of the column.
#         - The function assumes that the input column is numeric and contains valid data.
#         - If the column has zero standard deviation, the function will return NaN values.
#     """
#     norm_column = (column - mean(column)) / stdev(column)
#     return norm_column


# get the dataset
def get_dataset(path):
    """
    Load a dataset from a CSV file.

    Parameters:
        path (str): The path of the CSV file.

    Returns:
        tuple: A tuple containing the input features (X) and the target values (y).

    Example:
        >>> path = "data.csv"
        >>> X, y = get_dataset(path)
        >>> print(X.shape)
        (100, 2)
        >>> print(y.shape)
        (100, 2)

    """
    df = read_csv(path, header=None, skiprows=1)

    # split into input and output columns
    # X, y = df.values[:, 3:5], df.values[:, 1:3]
    X, y = df.values[:, 4:], df.values[:, 1:4]
    # X, y = np.concatenate([df.values[:, 3:4], df.values[:, 5:6]], axis=1), df.values[:, 1:3]
    #
    print("Xshape", X.shape, "yshape", y.shape)
    print(X.shape[1], y.shape[1])
    X = X.astype('float32')
    y = y.astype('float32')

    return X, y


# get the model
def generate_sequential_model(n_inputs, n_outputs, layers, capacity):
    """
    Create a sequential neural network model with the specified architecture.

    Parameters:
        n_inputs (int): The number of input features.
        n_outputs (int): The number of output features.
        layers (int): The number of hidden layers in the model.
        capacity (int): The capacity of the model, which affects the number of units in each layer.

    Returns:
        keras.Sequential: The constructed sequential neural network model.

    Description:
        This function creates a sequential neural network model with the specified architecture. The model consists of
        a specified number of hidden layers, each with the specified capacity (number of units). The input layer has
        the number of units equal to the number of input features. The output layer has the number of units equal to
        the number of output features. The activation function used in the hidden layers is ReLU, and the model is
        compiled with mean absolute error (MAE) loss and the Adam optimizer.

    Example:
        >>> n_inputs = 10
        >>> n_outputs = 1
        >>> layers = 3
        >>> capacity = 32
        >>> model = generate_sequential_model(n_inputs, n_outputs, layers, capacity)
        >>> print(model.summary())
        _________________________________________________________________
        Layer (type)                 Output Shape              Param #
        =================================================================
        dense_1 (Dense)              (None, 32)                352
        _________________________________________________________________
        dense_2 (Dense)              (None, 32)                1056
        _________________________________________________________________
        dense_3 (Dense)              (None, 32)                1056
        _________________________________________________________________
        dense_4 (Dense)              (None, 1)                 33
        =================================================================
        Total params: 2,497
        Trainable params: 2,497
        Non-trainable params: 0
        _________________________________________________________________

    Notes:
        - The model architecture can be adjusted by changing the number of layers and the capacity.
        - The activation function used in the hidden layers is ReLU.
        - The model is compiled with mean absolute error (MAE) loss and the Adam optimizer.
    """
    # Set the GPU device
    device_name = '/device:GPU:0'  # Use GPU device 0
    # device_name = '/gpu:0'  # Alternative syntax

    # Specify the device placement for the model
    with tf.device(device_name):
        model = Sequential()
        for i in range(0, layers):
            model.add(Dense(capacity, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))

        model.add(Activation('relu'))

        model.add(Dense(n_outputs))
        model.compile(loss='mae', optimizer='adam')

    return model


def get_sequential_other_model(n_inputs, n_outputs, layers, capacity):
    model = Sequential()

    for i in range(0, layers):
        if i == 0:
            model.add(Dense(capacity, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
        else:
            model.add(Dense(capacity, kernel_initializer='he_uniform', activation='relu'))

    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(Dense(n_outputs))
    model.compile(loss='mae', optimizer='adam')
    return model



def train_val_test_split(X, y, val_size=0.2, test_size=0.1, random_state=1, split2_only=False):
    """
    Split the dataset into training, validation, and test sets.

    Parameters:
        X (array-like): The input features.
        y (array-like): The target values.
        val_size (float): The proportion of the dataset to include in the validation set.
        test_size (float): The proportion of the dataset to include in the test set.
        random_state (int): The seed used by the random number generator.
        split2_only (bool): If True, only split into train+validation set and test set.

    Returns:
        tuple or tuple of tuples: A tuple or tuple of tuples containing the split datasets.

    Example:
        >>> X = [[1, 2], [3, 4], [5, 6]]
        >>> y = [0, 1, 0]
        >>> X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y)
        >>> print(X_train)
        [[5, 6]]
        >>> print(X_val)
        [[1, 2]]
        >>> print(X_test)
        [[3, 4]]
        >>> print(y_train)
        [0]
        >>> print(y_val)
        [0]
        >>> print(y_test)
        [1]

    """
    # First, split the data into train+validation set and test set
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)
    if split2_only:
        return X_train_val, X_test, y_train_val, y_test

    # Next, split train+validation set into training and validation sets
    val_adjusted = val_size / (1 - test_size)  # Adjust validation set size
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_adjusted, random_state=random_state)

    return X_train, X_val, X_test, y_train, y_val, y_test


# evaluate a model using repeated k-fold cross-validation
def evaluate_model(X, y, verbose, layers, capacity, epochs, patience, best_model):
    """
    Evaluate a sequential neural network model on the given data.

    Parameters:
        X (array-like): The input features.
        y (array-like): The target variable.
        verbose (int): The verbosity level of the training process.
        layers (int): The configuration of the neural network layers.
        capacity (int): The capacity of the model, which affects the number of units in each layer.
        epochs (int): The number of epochs to train the model.
        patience (int): The number of epochs to wait for improvement in validation loss before early stopping.
        best_model (str): The file path to save the best model.

    Returns:
        results (list): The evaluation results on the test set.

        model (keras.Sequential): The trained sequential neural network model.

        history (keras.callbacks.History): The training history containing the loss and metrics per epoch.

    Description:
        This function evaluates a sequential neural network model on the given data. The data is split into training,
        validation, and test sets using the `train_val_test_split()` function. The model is trained on the training
        set and evaluated on the test set. Early stopping is applied based on the validation loss with a specified
        patience value. The best model according to the validation loss is saved to the specified file path.

        The model architecture is defined using the `get_sequential_model()` function, which constructs a sequential
        model with the specified number of layers and capacity. The model is trained using the Adam optimizer and
        mean absolute error (MAE) loss.

    Example:
        >>> X = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        >>> y = [0, 1, 0]
        >>> verbose = 1
        >>> layers = 5
        >>> capacity = 32
        >>> epochs = 100
        >>> patience = 10
        >>> best_model = 'best_model.h5'
        >>> results, model, history = evaluate_model(X, y, verbose, layers, capacity, epochs, patience, best_model)
        >>> print(results)
        [0.2]
        >>> print(model)
        <tensorflow.python.keras.engine.sequential.Sequential object at 0x...>
        >>> print(history)
        <tensorflow.python.keras.callbacks.History object at 0x...>

    Notes:
        - The data is split into training, validation, and test sets using the `train_val_test_split()` function.
        - The model architecture is defined using the `get_sequential_model()` function.
        - The model is trained using the Adam optimizer and mean absolute error (MAE) loss.
        - Early stopping is applied based on the validation loss with a specified patience value.
        - The best model according to the validation loss is saved to the specified file path.
    """
    results = []
    n_inputs, n_outputs = X.shape[1], y.shape[1]
    print("X shape", n_inputs, "y", n_outputs)

    # Split data into training, validation, and test sets
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y, val_size=0.1, test_size=0.2)

    model = generate_sequential_model(n_inputs, n_outputs, layers, capacity)

    early_stopping = EarlyStopping(monitor='val_loss', patience=int(patience), verbose=1, mode='min',
                                   restore_best_weights=True)
    checkpoint = ModelCheckpoint(best_model, monitor='val_loss', save_best_only=True)

    # fit model
    model.fit(X_train, y_train, validation_data=(X_val, y_val), verbose=int(verbose),
              epochs=int(epochs), batch_size=64, validation_batch_size=64, callbacks=[early_stopping, checkpoint])
    history = model.history

    # evaluate model on test set
    mae = model.evaluate(X_test, y_test, verbose=1)
    results.append(mae)

    # # define evaluation procedure
    # cv = RepeatedKFold(n_splits=3, n_repeats=1, random_state=1)
    #
    # # enumerate folds
    # for train_ix, test_ix in cv.split(X_train):
    #     x_train_fold, x_test_fold = X_train[train_ix], X_train[test_ix]
    #     y_train_fold, y_test_fold = y_train[train_ix], y_train[test_ix]
    #
    #     # define model
    #     model = get_sequential_model(n_inputs, n_outputs, layers, capacity)
    #
    #     # model = get_sequential_other_model(n_inputs, n_outputs, layers, capacity)
    #
    #     early_stopping = EarlyStopping(monitor='val_loss', patience=int(patience), verbose=1, mode='min', restore_best_weights=True)
    #     checkpoint = ModelCheckpoint(best_model, monitor='val_loss', save_best_only=True)
    #     """
    #     When you use the ModelCheckpoint callback to save the best model during training, it saves the entire model to disk, including the weights, the architecture, the optimizer state, and any other model state that was present at the end of the epoch that had the best performance according to your specified metric. This is very useful because it allows you to resume training from that point if necessary, or to load the model for evaluation or prediction later without having to retrain.
    #
    #     On the other hand, the restore_best_weights argument of the EarlyStopping callback is slightly different. When this argument is set to True, the weights from the epoch with the best performance are stored in memory and then reloaded into the model when training is stopped. This means that after training ends, your model's weights are immediately the best ones observed during training, without you needing to manually load them from disk.
    #
    #     If you are using ModelCheckpoint to save the best model, and you are certain that you will manually load this model later when you need it, then you may not need to use restore_best_weights in EarlyStopping. However, if you want to use or evaluate the model immediately after training, restore_best_weights=True can be convenient because it saves you the step of loading the best weights from disk.
    #     """
    #     # fit model
    #     model.fit(x_train_fold, y_train_fold, validation_data=(x_test_fold, y_test_fold), verbose=int(verbose),
    #               epochs=int(epochs), batch_size=64, validation_batch_size=64, callbacks=[early_stopping, checkpoint])
    #     history = model.history
    #
    #     # evaluate model on test set
    #     mae = model.evaluate(X_test, y_test, verbose=1)
    #     results.append(mae)

    return results, model, history


def cal_result(results):
    return mean(results), std(results)


def plot_learning_curves(history, results, layers, capacity, patience, save_path, start, squared=False):
    """
    Plot the learning curves for a given model.

    Parameters:
        history (keras.callbacks.History): The training history of the model.
        results (list): The evaluation results of the model.
        layers (int): The number of layers in the model.
        capacity (int): The capacity of each layer in the model.
        patience (int): The patience value used for early stopping.
        save_path (str): The path to save the plot.
        start (int): The starting epoch for plotting.
        squared (bool): Whether to square the loss values.

    Returns:
        None

    Example:
        >>> history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))
        >>> results = [0.1, 0.2, 0.3]
        >>> layers = 3
        >>> capacity = 64
        >>> patience = 5
        >>> save_path = "learning_curves.png"
        >>> start = 1
        >>> plot_learning_curves(history, results, layers, capacity, patience, save_path, start, squared=True)

    """
    # Get training and validation loss values
    training_loss = history.history['loss'][int(start):]
    validation_loss = history.history['val_loss'][int(start):]

    if squared:
        training_loss = np.square(training_loss)
        validation_loss = np.square(validation_loss)

    # Create a larger figure
    plt.figure(figsize=(16, 9))

    # Create x-axis values (epochs)
    epochs = range(int(start), len(training_loss) + int(start))

    # Plot training and validation loss
    plt.plot(epochs, training_loss, 'bo--', label='Training Loss', markersize=0.5, linewidth=0.5, alpha=0.4)
    plt.plot(epochs, validation_loss, 'go--', label='Validation Loss', markersize=0.5, linewidth=0.5, alpha=0.4)

    # Add fit lines
    train_fit = np.polyfit(epochs, training_loss, 10)
    val_fit = np.polyfit(epochs, validation_loss, 10)
    train_fit_line = np.poly1d(train_fit)
    val_fit_line = np.poly1d(val_fit)
    plt.plot(epochs, train_fit_line(epochs), 'r--', alpha=0.8, label='Train Fit Line')
    plt.plot(epochs, val_fit_line(epochs), 'y--', alpha=0.8, label='Validation Fit Line')

    # Add labels and title
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Learning Curves')

    # Adjust legend position to top right
    plt.legend(loc='upper right')

    mean, std = cal_result(results)
    if squared:
        mean_s = np.square(mean * 1000)
        std_s = np.square(std * 1000)
    else:
        mean_s = 0
        std_s = 0
    # Add notes
    plt.annotate(f"MAE: mean={mean:.9f}, std={std:.9f} ms={mean_s:.6f}, ss={std_s:.6f}", xy=(0.03, 0.95),
                 xytext=(0.03, 0.95), xycoords='axes fraction', fontsize=12)
    plt.annotate(f"Layers: {layers}", xy=(0.03, 0.9), xytext=(0.03, 0.9), xycoords='axes fraction', fontsize=12)
    plt.annotate(f"Layer capacity: {capacity}", xy=(0.03, 0.85), xytext=(0.03, 0.85), xycoords='axes fraction',
                 fontsize=12)
    plt.annotate(f"Epochs: {epochs}", xy=(0.03, 0.8), xytext=(0.03, 0.8), xycoords='axes fraction', fontsize=12)
    plt.annotate(f"Patience: {patience}", xy=(0.03, 0.75), xytext=(0.03, 0.75), xycoords='axes fraction', fontsize=12)
    plt.annotate(f"squared: {squared}", xy=(0.03, 0.7), xytext=(0.03, 0.7), xycoords='axes fraction', fontsize=12)

    plt.savefig(save_path, dpi=1000)
    # Show the plot
    plt.show()


def write_predicted_y_analysed_to_csv(output_csv_path, predicted_y, new_y):
    if new_y.shape[1] == predicted_y.shape[1]:
        num_columns = new_y.shape[1]
    else:
        return "Error: the test file has different format with the predicted data"
    dif_y = predicted_y - new_y
    percent_y = dif_y / new_y
    percent_y_formatted = np.array(["{:.2f}%".format(value * 100) for value in percent_y.flat])
    percent_y_formatted = percent_y_formatted.reshape(percent_y.shape)

    empty_column = np.full((len(predicted_y), 1), "", dtype=object)

    combined_data = np.concatenate(
        (predicted_y, empty_column, new_y, empty_column, dif_y, empty_column, percent_y_formatted),
        axis=1)
    empty_header = ''
    for i in range(0, num_columns):
        empty_header += f", ''"

    headers = ['predicted_y', empty_header, 'new_y', empty_header, 'dif_y', empty_header, 'percent_y']
    header_row = ",".join(headers)
    with open(output_csv_path, 'w') as file:
        file.write(header_row + "\n")
        for row in combined_data:
            formatted_row = ",".join([f"{value:.18e}" if isinstance(value, (int, float)) else value for value in row])
            file.write(formatted_row + "\n")


def write_predicted_y_to_csv(output_csv_path, predicted_y):
    """
    Write the predicted values to a CSV file.

    Parameters:
        output_csv_path (str): The path to save the output CSV file.
        predicted_y (numpy.ndarray): The predicted values.

    Returns:
        None

    Example:
        >>> predicted_y = np.array([1.2, 2.3, 3.4])
        >>> output_csv_path = "predicted_values.csv"
        >>> write_predicted_y_to_csv(output_csv_path, predicted_y)

    """
    with open(output_csv_path, 'w') as file:
        for row in predicted_y:
            formatted_row = ",".join("{:.18e}".format(value) for value in row)
            file.write(formatted_row + "\n")


def generate_predicted_analysis_csv_paths(model_path, new_data_path, old_data=''):
    """
    Generate the paths for the predicted and analysis CSV files.

    Parameters:
        model_path (str): The path to the model file.
        new_data_path (str): The path to the new data file.
        old_data (str): Optional path to the old data file.

    Returns:
        predicted_path (str): The path for the predicted CSV file.
        analysis_path (str): The path for the analysis CSV file.

    Example:
        >>> model_path = "model.h5"
        >>> new_data_path = "new_data.csv"
        >>> predicted_path, analysis_path = generate_predicted_analysis_csv_paths(model_path, new_data_path)

    """

    # Create the directory if it doesn't exist
    if not os.path.exists('PredictedData'):
        os.makedirs('PredictedData')

    suffix = 1

    model_name = get_file_name(model_path)
    new_data = get_file_name(new_data_path)

    directory = f"M_{model_name}_D_{new_data}_{suffix}"
    if not os.path.exists(f'PredictedData\\{directory}'):
        os.makedirs(f'PredictedData\\{directory}')

    # Determine the file name
    predicted_path = os.path.join('PredictedData', directory, f"M_{model_name}_D_{new_data}_Pre{suffix}.csv")
    analysis_path = os.path.join('PredictedData', directory, f"M_{model_name}_D_{new_data}_Ana{suffix}.csv")

    while os.path.exists(predicted_path) or os.path.exists(analysis_path):
        suffix += 1
        predicted_path = os.path.join('PredictedData', directory, f"M_{model_name}_D_{new_data}_Pre{suffix}.csv")
        analysis_path = os.path.join('PredictedData', directory, f"M_{model_name}_D_{new_data}_Ana{suffix}.csv")

    return predicted_path, analysis_path


def generate_train_test_csv_path_from(file_path):
    """
    Generate the paths for the train and test CSV files.

    Parameters:
        file_path (str): The path to the original CSV file.

    Returns:
        output_train (str): The path for the train CSV file.
        output_test (str): The path for the test CSV file.

    Example:
        >>> file_path = "data.csv"
        >>> output_train, output_test = generate_train_test_csv_path_from(file_path)

    """
    # Create the directory if it doesn't exist
    if not os.path.exists('csv_test'):
        os.makedirs('csv_test')

    file_name = get_file_name(file_path)

    # Determine the file name
    suffix = 1
    output_train = os.path.join('csv_test', f"{file_name}_train_{suffix}.csv")
    output_test = os.path.join('csv_test', f"{file_name}_test_{suffix}.csv")

    while os.path.exists(output_train) or os.path.exists(output_test):
        suffix += 1
        output_train = os.path.join('csv_test', f"{file_name}_train_{suffix}.csv")
        output_test = os.path.join('csv_test', f"{file_name}_test_{suffix}.csv")

    return output_train, output_test


def generate_train_test_csvs_files_from(file_name, not_print=False):
    """
        Generate and save the training and test CSV files from the original file.

        Parameters:
            file_name (str): The path to the original file.
            not_print (bool): Flag to control whether to print the output file paths. Default is False.

        Returns:
            train_path (str): The path of the generated training CSV file.
            test_path (str): The path of the generated test CSV file.

        Example:
            >>> file_name = "data.csv"
            >>> train_path, test_path = generate_train_test_csvs_files_from(file_name)

        Notes:
            - This function uses the `train_test_split` function from scikit-learn to split the dataset into a training set and a test set.
            - The `test_size` parameter in `train_test_split` controls the proportion of the dataset that will be allocated to the test set. In this case, 20% of the data is allocated to the test set.
            - The `random_state` parameter ensures reproducibility of the train-test split.
            - The function calls the `generate_train_test_csv_path_from` function to generate the file paths for the train and test files.
            - The training and test sets are written to the respective CSV files using the `to_csv` method of pandas DataFrame.
            - If the `not_print` flag is set to True, the function does not print the output file paths.
    """
    from sklearn.model_selection import train_test_split
    # Load the dataset
    df = read_csv(file_name)

    # Split the dataset into training set and test set
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=1)

    train_path, test_path = generate_train_test_csv_path_from(file_name)

    # Write the training set and test set to CSV files
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    if not_print:
        print("output train: ", train_path)
        print("output test: ", test_path)

    return train_path, test_path


def generate_PC_csv_path_from(file_path):
    """
    Generate and return the path for the principal component (PC) CSV file.

    Parameters:
        file_path (str): The path to the original file.

    Returns:
        pc_path (str): The path of the generated PC CSV file.

    Example:
        >>> file_path = "data.csv"
        >>> pc_path = generate_PC_csv_path_from(file_path)

    Notes:
        - This function generates the path for the PC CSV file based on the original file path.
        - The PC CSV file is stored in the 'csv_test' directory.
        - The file name of the PC CSV file includes the original file name and a suffix indicating the version.
        - The suffix is incremented if a file with the same name already exists.
    """

    directory = 'PCA Models and PCs csv'
    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    file_name = get_file_name(file_path)

    # Determine the file name
    suffix = 1
    pc_path = os.path.join(directory, f"{file_name}_pc_{suffix}.csv")

    while os.path.exists(pc_path):
        suffix += 1
        pc_path = os.path.join(directory, f"{file_name}_pc_{suffix}.csv")

    return pc_path


# TODO:Finish the SAVING PC DATA
def generate_PC_csv_file_from(file_path, columns):
    df = read_csv(file_path)

    # selecting the 2nd and 4th columns (Python uses 0-based indexing)
    df_new = df.iloc[:, columns]

    pc_path = generate_PC_csv_path_from(file_path)

    # Save the new dataframe to a csv file
    df_new.to_csv(pc_path, index=False)

    return pc_path




def generate_sequential_model_and_curve_path(data, epochs, layers, capacity, patience, squared):
    """
    Generate and return the paths for the sequential model and learning curve image.

    Parameters:
        data (str): The path to the data file.
        epochs (int): The number of epochs for training the model.
        layers (int): The number of layers in the sequential model.
        capacity (int): The capacity of each layer in the sequential model.
        patience (int): The patience value for early stopping during training.
        squared (bool): Whether to square the loss values for the learning curve.

    Returns:
        model_path (str): The path for the sequential model file.
        curve_path (str): The path for the learning curve image.

    Example:
        >>> data = "data.csv"
        >>> epochs = 100
        >>> layers = 3
        >>> capacity = 64
        >>> patience = 5
        >>> squared = False
        >>> model_path, curve_path = generate_sequential_model_and_curve_path(data, epochs, layers, capacity, patience, squared)

    Notes:
        - This function generates the paths for the sequential model and learning curve image based on the provided parameters.
        - The sequential model file is stored in the 'Models' directory.
        - The model name includes the data name, number of epochs, number of layers, capacity, and patience.
        - The learning curve image file name includes the data name, suffix, number of epochs, number of layers, patience, and squared flag.
        - The suffix is incremented if a file with the same name already exists.
    """
    # Create the directory if it doesn't exist
    if not os.path.exists('Models'):
        os.makedirs('Models')

    data_name = get_file_name(data)

    # Determine the model name
    suffix = 1
    model_path = os.path.join("Models", f"{data_name}_{suffix}el{int(epochs)}{int(layers)}pat{patience}_c{capacity}")
    curve_path = os.path.join('Models', f"{data_name}_loss{suffix}el{int(epochs)}{int(layers)}pat{patience}s{squared}.png")

    while os.path.exists(model_path):
        suffix += 1
        model_path = os.path.join("Models", f"{data_name}_{suffix}el{int(epochs)}{int(layers)}pat{patience}_c{capacity}")
        curve_path = os.path.join('Models',
                                    f"{data_name}_loss{suffix}el{int(epochs)}{int(layers)}pat{patience}s{squared}.png")

    return model_path, curve_path




# *****************************PLOTING FUNCIONS********************************** #
# *****************************PLOTING FUNCIONS********************************** #
# *****************************PLOTING FUNCIONS********************************** #


def get_data_sole(file_path, column):
    """
    Extract and return the numeric values from a specific column in a CSV file.

    Parameters:
        file_path (str): The path to the CSV file.
        column (str): The column letter in the file.

    Returns:
        data_numeric (Series): The numeric values from the specified column.

    Example:
        >>> file_path = "data.csv"
        >>> column = "B"
        >>> data = get_data_sole(file_path, column)

    Notes:
        - This function reads the CSV file located at 'file_path' and extracts the values from the column specified by 'column'.
        - The column letter should be specified using uppercase letters ('A' for the first column, 'B' for the second column, and so on).
        - If the column contains percentage values (e.g., "25%"), the function converts the values to numeric type (e.g., 0.25).
        - If the column does not contain percentage values, the function returns the values as they are.
    """
    numeric_value = ord(column) - ord('A')
    file = read_csv(file_path)
    data = file.iloc[:, numeric_value]

    if '%' in str(data):
        # Remove the percentage symbol (%) and convert to numeric type
        data_numeric = data.str.rstrip('%').astype(float) / 100.0
    else:
        # Column does not contain percentage values, handle accordingly
        data_numeric = data

    return data_numeric


def get_rid_of_edge(data_1, data_2, data_3, threshold_fraction, dif_1, dif_2, dif_3, p_1, p_2, p_3):
    threshold = threshold_fraction

    # Calculate the 10th and 90th percentiles for each coordinate
    lower_percentile_1 = data_1.quantile(threshold / 2)
    upper_percentile_1 = data_1.quantile(1 - (threshold / 2))
    print("data1", threshold / 2, 1 - threshold / 2, lower_percentile_1, upper_percentile_1)

    lower_percentile_2 = data_2.quantile(threshold / 2)
    upper_percentile_2 = data_2.quantile(1 - (threshold / 2))
    print("data2", lower_percentile_2, upper_percentile_2)

    lower_percentile_3 = data_3.quantile(threshold / 2)
    upper_percentile_3 = data_3.quantile(1 - (threshold / 2))
    print("data3", lower_percentile_3, upper_percentile_3)

    # Filter the data based on percentiles for each coordinate
    filtered_data_1 = data_1[(data_1 >= lower_percentile_1) & (data_1 <= upper_percentile_1)]

    filtered_data_2 = data_2[(data_2 >= lower_percentile_2) & (data_2 <= upper_percentile_2)]

    filtered_data_3 = data_3[(data_3 >= lower_percentile_3) & (data_3 <= upper_percentile_3)]

    # Get the joined index of all three data sets
    common_index = filtered_data_1.index.intersection(filtered_data_2.index).intersection(filtered_data_3.index)

    filtered_data_1 = filtered_data_1.loc[common_index]
    filtered_data_2 = filtered_data_2.loc[common_index]
    filtered_data_3 = filtered_data_3.loc[common_index]

    filtered_dif_1 = dif_1.loc[common_index]
    filtered_dif_2 = dif_2.loc[common_index]
    filtered_dif_3 = dif_3.loc[common_index]

    filtered_p_1 = p_1.loc[common_index]
    filtered_p_2 = p_2.loc[common_index]
    filtered_p_3 = p_3.loc[common_index]

    return filtered_data_1, filtered_data_2, filtered_data_3, filtered_dif_1, filtered_dif_2, filtered_dif_3, filtered_p_1, filtered_p_2, filtered_p_3


def plot_hist(data, save_path, bin, square, type):
    """
    Plot a histogram of the given data.

    Parameters:
        data (array-like): The data to plot the histogram for.
        save_path (str): The path to save the generated plot.
        bin (int or array-like): The number of bins to use for the histogram or the bin edges.
        square (bool): Indicates whether to square the data values before plotting.
        type (str): The type or name of the data.

    Returns:
        plt (matplotlib.pyplot): The matplotlib pyplot object.

    Example:
        >>> plot_hist(data, save_path, bin=10, square=True, type="Y1")

    Notes:
        - This function plots a histogram of the given data using the specified number of bins.
        - The 'data' parameter should be an array-like object containing the data values.
        - The 'save_path' parameter should be a string specifying the path to save the generated plot.
        - The 'bin' parameter can be an integer specifying the number of bins to use or an array-like object specifying the bin edges.
        - The 'square' parameter indicates whether to square the data values before plotting. Set it to True or False accordingly.
        - The 'type' parameter specifies the type or name of the data being plotted.
        - The function saves the plot as an image file at the specified save path and returns the matplotlib.pyplot object.
    """
    plt.figure(figsize=(16, 9))

    plt.hist(data, bin, edgecolor='blue')

    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(f"Histogram for {type}")

    plt.annotate(f"bin = {bin}", xy=(0.9, 0.95), xytext=(0.9, 0.95), xycoords='axes fraction', fontsize=12)
    plt.annotate(f"type = {type}", xy=(0.9, 0.90), xytext=(0.9, 0.90), xycoords='axes fraction', fontsize=12)
    plt.annotate(f"squared: {square}", xy=(0.9, 0.85), xytext=(0.9, 0.85), xycoords='axes fraction', fontsize=12)

    plt.savefig(save_path, dpi=800)

    plt.show()

    return plt


def generate_hist_png_path(file_name, bin, type, square):
    """
    Generate the file path for saving the histogram plot as a PNG image.

    Parameters:
        file_name (str): The base file name or identifier for the histogram plot.
        bin (int or array-like): The number of bins used for the histogram or the bin edges.
        type (str): The type or name of the data being plotted.
        square (bool): Indicates whether the data values are squared.

    Returns:
        output_file (str): The file path for saving the histogram plot.

    Example:
        >>> generate_hist_png_path(file_name, bin=10, type="Data", square=True)

    Notes:
        - This function generates a unique file path for saving the histogram plot as a PNG image.
        - The 'file_name' parameter should be a string representing the base file name or identifier for the plot.
        - The 'bin' parameter can be an integer specifying the number of bins used or an array-like object specifying the bin edges.
        - The 'type' parameter specifies the type or name of the data being plotted.
        - The 'square' parameter indicates whether the data values are squared.
        - The function appends suffixes to the file name to ensure uniqueness in case of existing files with the same name.
        - The function returns the generated file path for saving the histogram plot.
    """
    # determine the file name
    suffix = 1
    output_file = f"{str(file_name)}_HIST_type={str(type)}_bin={int(bin)}_squared={square}_{suffix}.png"
    while os.path.exists(output_file):
        suffix += 1
        output_file = f"{str(file_name)}_HIST_type={str(type)}_bin={int(bin)}_squared={square}_{suffix}.png"

    return output_file


def get_data_corresponding(file_name, column):

    original_numeric_value = ord(column) - ord('A')

    dif_numeric_value = original_numeric_value + 4
    percent_numeric_value = original_numeric_value + 8

    file = read_csv(file_name, skiprows=1)
    data_1 = file.iloc[:, original_numeric_value]
    data_2 = file.iloc[:, original_numeric_value + 1]
    data_3 = file.iloc[:, original_numeric_value + 2]
    print("data_1", data_1)
    data_dif_1 = file.iloc[:, dif_numeric_value]
    data_dif_2 = file.iloc[:, dif_numeric_value + 1]
    data_dif_3 = file.iloc[:, dif_numeric_value + 2]

    data_percent_1 = file.iloc[:, percent_numeric_value]
    data_percent_2 = file.iloc[:, percent_numeric_value + 1]
    data_percent_3 = file.iloc[:, percent_numeric_value + 2]

    # Remove the percentage symbol (%) and convert to numeric type
    data_percent_1 = data_percent_1.str.rstrip('%').astype(float) / 100.0
    data_percent_2 = data_percent_2.str.rstrip('%').astype(float) / 100.0
    data_percent_3 = data_percent_3.str.rstrip('%').astype(float) / 100.0

    # return data_1, data_2, data_z, data_dif_1, data_dif_y, data_dif_z
    return data_1, data_2, data_3, data_dif_1, data_dif_2, data_dif_3, data_percent_1, data_percent_2, data_percent_3


def plot_scatter(save_path, data, data_dif, value, type, threshold):
    """
    Create a scatter plot of two data columns.

    Parameters:
        save_path (str): The file path to save the plot.
        data (Series): The data column for the x-axis.
        data_dif (Series): The data column for the y-axis.
        value (str): The label for the x-axis.
        type (str): The label for the y-axis.
        threshold (float): The threshold value for filtering the data.

    Returns:
        None

    Example:
        >>> plot_scatter(save_path, data, data_dif, value='X', type='Difference', threshold=0.2)

    Notes:
        - This function creates a scatter plot of two data columns.
        - The 'save_path' parameter should be a string representing the file path to save the plot.
        - The 'data' and 'data_dif' parameters should be pandas Series objects representing the data columns.
        - The 'value' and 'type' parameters should be strings representing the labels for the x-axis and y-axis, respectively.
        - The 'threshold' parameter should be a float representing the threshold value for filtering the data.
        - The function saves the plot to the specified file path.
    """
    plt.figure(figsize=(16, 9))

    plt.scatter(data, data_dif)

    plt.xlabel(f"{value}")
    plt.ylabel(f"{value}-{type}")
    plt.title(f'Scatter Plot: {value} vs {value}-{type}')

    plt.annotate(f"filtered: {threshold * 100}%", xy=(0.85, 0.95), xytext=(0.85, 0.95), xycoords='axes fraction',
                 fontsize=14)

    plt.savefig(save_path, dpi=800)

    plt.show()


def generate_scatter_png_path(file_path, value, type, threshold):
    """
    Generate the file path for saving a scatter plot.

    Parameters:
        file_path (str): The file path of the data file.
        value (str): The label for the x-axis.
        type (str): The label for the y-axis.
        threshold (float): The threshold value for filtering the data.

    Returns:
        str: The file path for saving the scatter plot.

    Example:
        >>> generate_scatter_png_path(file_path, value='X', type='Difference', threshold=0.2)

    Notes:
        - This function generates a file path for saving a scatter plot based on the input parameters.
        - The 'file_path' parameter should be a string representing the file path of the data file.
        - The 'value' and 'type' parameters should be strings representing the labels for the x-axis and y-axis, respectively.
        - The 'threshold' parameter should be a float representing the threshold value for filtering the data.
        - The function appends relevant information to the file name to avoid overwriting existing files.
        - The returned file path can be used to save the scatter plot.
    """
    file_name = get_file_name(file_path)

    threshold_percentage = int(threshold * 100)

    # determine the file name
    suffix = 1
    plot_path = f"{str(file_name)}_SCATTER_{str(type)}_{str(value)}_filter={threshold_percentage}%_{suffix}.png"
    while os.path.exists(plot_path):
        suffix += 1
        plot_path = f"{str(file_name)}_SCATTER_{str(type)}_{str(value)}_filter={threshold_percentage}%_{suffix}.png"

    return plot_path


def plot_2_outputs(data_1, data_2, data_3, threshold_fraction, data_dif_1, data_dif_2, data_dif_3, data_p_1, data_p_2, data_p_3, file_path, type_dif, type_percent, directory):
    """
    Plot scatter plots for two outputs.

    Parameters:
        data_1 (pd.Series): Data for output 1.
        data_2 (pd.Series): Data for output 2.
        threshold_fraction (float): Threshold fraction for filtering the data.
        data_dif_1 (pd.Series): Difference data for output 1.
        data_dif_2 (pd.Series): Difference data for output 2.
        data_p_1 (pd.Series): Percentage data for output 1.
        data_p_2 (pd.Series): Percentage data for output 2.
        file_path (str): File path of the data file.
        type_dif (str): Label for the difference type.
        type_percent (str): Label for the percentage type.
        directory (str): Directory path for saving the scatter plots.

    Returns:
        None

    Notes:
        - This function plots scatter plots for two outputs based on the input data and parameters.
        - The 'data_1' and 'data_2' parameters should be pandas Series representing the data for output 1 and output 2, respectively.
        - The 'threshold_fraction' parameter should be a float representing the threshold fraction for filtering the data.
        - The 'data_dif_1' and 'data_dif_2' parameters should be pandas Series representing the difference data for output 1 and output 2, respectively.
        - The 'data_p_1' and 'data_p_2' parameters should be pandas Series representing the percentage data for output 1 and output 2, respectively.
        - The 'file_path' parameter should be a string representing the file path of the data file.
        - The 'type_dif' and 'type_percent' parameters should be strings representing the labels for the difference type and percentage type, respectively.
        - The 'directory' parameter should be a string representing the directory path for saving the scatter plots.
        - The function removes data points at the edges based on the 'threshold_fraction' parameter before plotting the scatter plots.
        - The scatter plots are saved in the specified directory.
    """
    data_1, data_2, data_3, data_dif_1, data_dif_2, data_dif_3, data_p_1, data_p_2. data_p_3 = \
        get_rid_of_edge(data_1, data_2, data_3, threshold_fraction, data_dif_1, data_dif_2, data_dif_3, data_p_1, data_p_2, data_p_3)

    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    save_path_d1 = os.path.join(directory, generate_scatter_png_path(file_path, "y1", type_dif, threshold_fraction))
    print(save_path_d1)
    plot_scatter(save_path_d1, data_1, data_dif_1, "y1", type_dif, threshold_fraction)

    save_path_p1 = os.path.join(directory, generate_scatter_png_path(file_path, "y1", type_percent, threshold_fraction))
    print(save_path_p1)
    plot_scatter(save_path_p1, data_1, data_p_1, "y1", type_percent, threshold_fraction)

    save_path_d2 = os.path.join(directory, generate_scatter_png_path(file_path, "y2", type_dif, threshold_fraction))
    print(save_path_d2)
    plot_scatter(save_path_d2, data_2, data_dif_2, "y2", type_dif, threshold_fraction)

    save_path_p2 = os.path.join(directory, generate_scatter_png_path(file_path, "y2", type_percent, threshold_fraction))
    print(save_path_p2)
    plot_scatter(save_path_p2, data_2, data_p_2, "y2", type_percent, threshold_fraction)

    save_path_d3 = os.path.join(directory, generate_scatter_png_path(file_path, "y3", type_dif, threshold_fraction))
    print(save_path_d3)
    plot_scatter(save_path_d3, data_3, data_dif_3, "y3", type_dif, threshold_fraction)

    save_path_p3 = os.path.join(directory, generate_scatter_png_path(file_path, "y3", type_percent, threshold_fraction))
    print(save_path_p3)
    plot_scatter(save_path_p3, data_3, data_p_3, "y3", type_percent, threshold_fraction)





# ***************************************** Learning And Predicting *********************************** #
# ***************************************** Learning And Predicting *********************************** #
# ***************************************** Learning And Predicting *********************************** #

def fit_model_save_best_and_curve(data, epochs, layers, capacity, patience, epochs_start, squared=False, print_path=False):
    # load dataset
    x, y = get_dataset(data)
    if print_path:
        print("Training input x: \n", x, "\nTraining output y: \n", y)

    set_up_gpu()

    model_path, learning_curve_path = generate_sequential_model_and_curve_path(data, epochs, layers, capacity, patience, squared)

    # evaluate model
    results, model, history = evaluate_model(x, y, 2, layers, capacity, epochs, patience, model_path)

    if print_path:
        print("Successfully trained and saved model! \nModel_path: ", model_path)

    # mean, std = cal_result(results)
    # # summarize performance
    # print('MAE: %.3f (%.3f)' % (mean(results), std(results)))

    plot_learning_curves(history, results, layers, capacity, patience, learning_curve_path, epochs_start, squared)
    if print_path:
        print("Successfully plotted learning curv! \nLearning curve path: ", learning_curve_path)

    return model_path, learning_curve_path


def load_model_to_predict_analysis_plot(model_path, data, old_data ='', type_dif="dif", type_percent="percent", threshold_start=0,
                                        threshold_step=0.05, threshold_range=5, ):
    loaded_model = load_model(model_path)

    new_x, new_y = get_dataset(data)
    print('new x=', new_x)

    # get the result of the predicted output y
    predicted_y = loaded_model.predict(new_x)


    print(predicted_y)

    predicted_path, analysis_path = generate_predicted_analysis_csv_paths(model_path, data, old_data)

    directory = os.path.dirname(predicted_path)

    print("output predicted path: ", predicted_path)
    print("output analysis path: ", analysis_path)

    write_predicted_y_to_csv(predicted_path, predicted_y)
    write_predicted_y_analysed_to_csv(analysis_path, predicted_y, new_y)

    data_1, data_2, data_3, data_dif_1, data_dif_2, data_dif_3, data_p_1, data_p_2, data_p_3 = get_data_corresponding(analysis_path, 'E')

    analysis_name = get_file_name(analysis_path)

    for i in range(0, threshold_range):
        plot_2_outputs(data_1, data_2, threshold_start, data_dif_1, data_dif_2, data_p_1, data_p_2, analysis_name, type_dif, type_percent, directory)
        threshold_start += threshold_step

    return directory, predicted_path, analysis_path

def machine_learning_save_predict(train_data, test_data, epochs=5000, layers=9, capacity=32, patience=500, epochs_start=1000):
    model_path, learning_curve_path = fit_model_save_best_and_curve(train_data, epochs, layers, capacity, patience, epochs_start)
    directory, predicted_path, analysis_path = load_model_to_predict_analysis_plot(model_path, test_data)
    print("Model Path: ", model_path, "\nPredicted Data: ", directory)

    return model_path, learning_curve_path, directory, predicted_path, analysis_path



