import tensorflow as tf
from dataset.Dataset import x_test


def representative_data_gen():
    """Generator function for the representative dataset required by the TFLite
    converter for quantization.

    Yields:
    ---------------
    list
        List containing a single batch of data. In this case, the batch size is 1.
    """
    for data in tf.data.Dataset.from_tensor_slices(x_test).batch(1).take(100):
        yield [(tf.dtypes.cast(data, tf.float32))]


def convert_to_tflite(keras_model, generation=0, i=0, time=0):
    """Convert a Keras model to a TFLite model and save it to disk.

    Parameters:
    ----------------
    keras_model : tf.keras.Model
        Keras model to be converted.

    generation : int, optional
        Generation number of the model, default is 0.

    i : int, optional
        Index of the model, default is 0.

    time : int, optional
        Time, default is 0.

    Returns:
    ----------------
    tflite_model : bytes
        The converted TFLite model in binary format.

    path : str
        Path of the saved TFLite model.
    """
    # Initialize a TFLite converter from the Keras model
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)

    # Enable optimization (i.e., quantization)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Set the representative dataset for quantization
    converter.representative_dataset = representative_data_gen

    # Explicitly declare the supported types for full integer quantization
    converter.target_spec.supported_types = [tf.int8]

    # Set the input and output tensors to uint8
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    # Allow custom operations
    converter.allow_custom_ops = True
    # Enable experimental converter
    converter.experimental_new_converter = True
    # Enable experimental new quantizer
    converter.experimental_new_quantizer = True

    # Convert the Keras model to a TFLite model
    tflite_model = converter.convert()

    # Define the path where to save the TFLite model
    path = f"model_{i}_gen_{generation}_time_{time}.tflite"

    # Save the TFLite model to disk
    with open(path, "wb") as f:
        f.write(tflite_model)

    # Clean up the converter
    del converter

    return tflite_model, path
