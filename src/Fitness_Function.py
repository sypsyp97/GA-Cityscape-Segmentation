# calculate the fitness
from math import pi

import numpy as np


def calculate_fitness(acc, iou, inference_time):
    """Calculate the fitness of a model based on its accuracy, intersection over union
    (IoU), and inference time. The fitness function is designed to prioritize models
    with higher accuracy and IoU, and lower inference time.

    Parameters:
    ----------------
    acc : float
        The accuracy of the model on a validation or test set.

    iou : float
        The intersection over union of the model's predictions on a validation or test set.

    inference_time : float
        The time taken by the model to make predictions on a validation or test set.

    Returns:
    ----------------
    fitness : float
        The calculated fitness of the model.
    """
    fitness = (1 - np.arctan(inference_time / 500) / (pi / 2)) * acc * iou

    return fitness
