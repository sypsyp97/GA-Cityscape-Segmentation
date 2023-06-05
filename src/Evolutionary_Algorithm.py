import gc

import numpy as np
from src.Create_Model import create_model
from src.Create_Model import train_model
from src.Model_Checker import model_has_problem
from src.TFLITE_Converter import convert_to_tflite
from src.Compile_Edge_TPU import compile_edgetpu
from src.Inference_Speed_TPU import inference_time_tpu
from src.Fitness_Function import calculate_fitness
import pickle
import os


def create_first_population(population, num_classes=34):
    """
    Create an initial population for a genetic algorithm.

    This function initializes the first population for the genetic algorithm. It
    creates a population array filled with binary values. Each individual in the
    population is evaluated with a model to ensure that the individual does not
    produce an error.

    Parameters:
    -------------
    population : int
        The size of the population.

    num_classes : int, default=34
        The number of classes to predict.

    Returns:
    -------------
    first_population_array : np.ndarray
        The array representing the first population.
    """

    first_population_array = np.random.randint(0, 2, (population, 9, 18))

    result_dir = f'arrays'
    array_dir = result_dir + '/first_population_array.pkl'

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    for i in range(population):
        model = create_model(first_population_array[i], num_classes=num_classes)
        while model_has_problem(model):
            del model
            first_population_array[i] = np.random.randint(0, 2, (9, 18))
            model = create_model(first_population_array[i], num_classes=num_classes)

    with open(array_dir, 'wb') as f:
        pickle.dump(first_population_array, f)

        del model

    return first_population_array


def select_models(train_ds,
                  val_ds,
                  test_ds,
                  time,
                  population_array,
                  generation,
                  epochs=30,
                  num_classes=34):
    """
    Select the best models from the population.

    This function evaluates each individual in the population by creating a model
    from the individual and training it. The performance of the model on the test set
    is used as the fitness of the individual. The top performing individuals are selected
    to create the next generation.

    Parameters:
    -------------
    train_ds : tf.data.Dataset
        The dataset for training.

    val_ds : tf.data.Dataset
        The dataset for validation.

    test_ds : tf.data.Dataset
        The dataset for testing.

    # ... other parameters ...

    Returns:
    -------------
    best_models_arrays : np.ndarray
        The array of the best performing individuals.

    max_fitness : float
        The highest fitness value.

    average_fitness : float
        The average fitness value.
    """
    fitness_list = []
    tflite_accuracy_list = []
    tpu_time_list = []
    iou_list = []

    result_dir = f'results_{time}'
    generation_dir = result_dir + f'/generation_{generation}'
    best_models_arrays_dir = generation_dir + '/best_model_arrays.pkl'
    fitness_list_dir = generation_dir + '/fitness_list.pkl'
    tflite_accuracy_list_dir = generation_dir + '/tflite_accuracy_list.pkl'
    iou_list_dir = generation_dir + '/iou_list.pkl'
    tpu_time_list_dir = generation_dir + '/tpu_time_list.pkl'

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    if not os.path.exists(generation_dir):
        os.makedirs(generation_dir)

    for i in range(population_array.shape[0]):
        model = create_model(population_array[i], num_classes=num_classes)
        model, history = train_model(train_ds, val_ds, model=model, epochs=epochs)
        _, tflite_accuracy, iou = model.evaluate(test_ds)
        try:
            tflite_model, tflite_name = convert_to_tflite(keras_model=model, generation=generation, i=i, time=time)
            edgetpu_name = compile_edgetpu(tflite_name)
            tpu_time = inference_time_tpu(edgetpu_model_name=edgetpu_name)
        except:
            tpu_time = 9999

        fitness = calculate_fitness(tflite_accuracy, iou, tpu_time)

        tflite_accuracy_list.append(tflite_accuracy)
        iou_list.append(iou)
        fitness_list.append(fitness)
        tpu_time_list.append(tpu_time)

        with open(fitness_list_dir, 'wb') as f:
            pickle.dump(fitness_list, f)
        with open(tflite_accuracy_list_dir, 'wb') as f:
            pickle.dump(tflite_accuracy_list, f)
        with open(iou_list_dir, 'wb') as f:
            pickle.dump(iou_list, f)
        with open(tpu_time_list_dir, 'wb') as f:
            pickle.dump(tpu_time_list, f)

        gc.collect()

    max_fitness = np.max(fitness_list)
    average_fitness = np.average(fitness_list)

    best_models_indices = sorted(range(len(fitness_list)), key=lambda j: fitness_list[j], reverse=True)[:5]
    best_models_arrays = [population_array[k] for k in best_models_indices]

    with open(best_models_arrays_dir, 'wb') as f:
        pickle.dump(best_models_arrays, f)

    return best_models_arrays, max_fitness, average_fitness


def crossover(parent_arrays):
    """
    Perform the crossover operation.

    This function generates a child individual from two parent individuals
    by randomly selecting each gene from one of the parents.

    Parameters:
    -------------
    parent_arrays : list
        The list of parent individuals.

    Returns:
    -------------
    child_array : np.ndarray
        The child individual.
    """
    parent_indices = np.random.randint(0, 5, size=parent_arrays[0].shape)
    child_array = np.choose(parent_indices, parent_arrays)
    return child_array


def mutate(model_array, mutate_prob=0.05):
    """
    Perform the mutation operation.

    This function mutates an individual by flipping the value of a gene with a
    certain probability.

    Parameters:
    -------------
    model_array : np.ndarray
        The individual to mutate.

    mutate_prob : float, default=0.05
        The probability of mutation.

    Returns:
    -------------
    mutated_array : np.ndarray
        The mutated individual.
    """
    prob = np.random.uniform(size=(9, 18))
    mutated_array = np.where(prob < mutate_prob, np.logical_not(model_array), model_array)

    return mutated_array


def create_next_population(parent_arrays, population=10, num_classes=34):
    """
    Create the next population.

    This function creates the next population by performing crossover and mutation
    operations on the best performing individuals from the previous population.

    Parameters:
    -------------
    parent_arrays : list
        The list of parent individuals.

    population : int, default=10
        The size of the population.

    num_classes : int, default=34
        The number of classes to predict.

    Returns:
    -------------
    next_population_array : np.ndarray
        The array representing the next population.
    """

    next_population_array = np.random.randint(0, 2, (population, 9, 18))

    for individual in range(population):
        next_population_array[individual] = crossover(parent_arrays)
        next_population_array[individual] = mutate(next_population_array[individual], mutate_prob=0.03)

    for individual in range(population):
        model = create_model(next_population_array[individual], num_classes=num_classes)
        while model_has_problem(model):
            del model
            next_population_array[individual] = crossover(parent_arrays)
            next_population_array[individual] = mutate(next_population_array[individual], mutate_prob=0.03)
            model = create_model(next_population_array[individual], num_classes=num_classes)
        del model

    return next_population_array


def start_evolution(train_ds, val_ds, test_ds, generations, population, num_classes, epochs, population_array=None,
                    time=None):
    """
    Run the genetic algorithm.

    This function executes the genetic algorithm for a certain number of generations.
    In each generation, it selects the best individuals, creates the next population,
    and records the best and average fitness values.

    Parameters:
    -------------
    train_ds : tf.data.Dataset
        The dataset for training.

        val_ds : tf.data.Dataset
        The dataset for validation.

    test_ds : tf.data.Dataset
        The dataset for testing.

    generations : int
        The number of generations to evolve.

    population : int
        The size of the population.

    num_classes : int
        The number of classes to predict.

    epochs : int
        The number of epochs to train each model.

    population_array : np.ndarray, default=None
        The array representing the initial population. If None, an initial population
        will be created.

    time : int, default=None
        A timestamp to use for naming the result directories.

    Returns:
    -------------
    population_array : np.ndarray
        The array representing the final population.

    max_fitness_history : list
        The history of the highest fitness value in each generation.

    average_fitness_history : list
        The history of the average fitness value in each generation.

    best_models_arrays : list
        The array of the best performing individuals from the final generation.
    """

    max_fitness_history = []
    average_fitness_history = []
    if population_array is None:
        population_array = create_first_population(population=population, num_classes=num_classes)

    result_dir = f'results_{time}'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    for generation in range(generations):
        best_models_arrays, max_fitness, average_fitness = select_models(train_ds=train_ds, val_ds=val_ds,
                                                                         test_ds=test_ds, time=time,
                                                                         population_array=population_array,
                                                                         generation=generation, epochs=epochs,
                                                                         num_classes=num_classes)
        population_array = create_next_population(parent_arrays=best_models_arrays, population=population,
                                                  num_classes=num_classes)
        max_fitness_history.append(max_fitness)
        average_fitness_history.append(average_fitness)

        next_population_array_dir = result_dir + '/next_population_array.pkl'
        max_fitness_history_dir = result_dir + '/max_fitness_history.pkl'
        average_fitness_history_dir = result_dir + '/average_fitness_history.pkl'
        best_model_arrays_dir = result_dir + '/best_model_arrays.pkl'

        with open(next_population_array_dir, 'wb') as f:
            pickle.dump(population_array, f)
        with open(max_fitness_history_dir, 'wb') as f:
            pickle.dump(max_fitness_history, f)
        with open(average_fitness_history_dir, 'wb') as f:
            pickle.dump(average_fitness_history, f)
        with open(best_model_arrays_dir, 'wb') as f:
            pickle.dump(best_models_arrays, f)

    return population_array, max_fitness_history, average_fitness_history, best_models_arrays
