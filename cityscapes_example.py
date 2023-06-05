import os
import pickle
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Sets the TensorFlow C++ minimum log level to ERROR
os.environ['PYTHONHASHSEED'] = "1234"  # Set PYTHONHASHSEED to enable deterministic behavior

from src.Evolutionary_Algorithm import start_evolution
from dataset.Dataset import train_ds, val_ds, test_ds
import tensorflow as tf
import gc  # Importing Python's garbage collector module
from datetime import datetime
import numpy as np
import random

random.seed(1234)  # Setting random seed for random package for reproducibility
tf.random.set_seed(1234)  # Setting random seed for TensorFlow for reproducibility
np.random.seed(1234)  # Setting random seed for numpy for reproducibility

# Load the initial population for the evolutionary algorithm from a pickle file
with open('results_10042023182244/next_population_array.pkl', 'rb') as f:
    data = pickle.load(f)
    f.close()


if __name__ == '__main__':
    gc.enable()  # Enable automatic garbage collection
    now = datetime.now()
    formatted_date = now.strftime("%d%m%Y%H%M%S")  # Get the current date and time

    print(train_ds, val_ds, test_ds)  # Print the training, validation, and test datasets

    # Call the start_evolution function to begin the evolutionary algorithm
    population_array, max_fitness_history, average_fitness_history, best_models_arrays = start_evolution(
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        generations=19,  # Number of generations for the evolutionary algorithm
        population=20,  # Size of population for the evolutionary algorithm
        num_classes=34,  # Number of output classes
        epochs=40,  # Number of epochs for training the models
        population_array=data,  # Initial population
        time=formatted_date)  # Current date and time

