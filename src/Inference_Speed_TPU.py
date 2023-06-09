import time

import numpy as np
from PIL import Image
from pycoral.utils.edgetpu import make_interpreter

image_file = "test.jpg"
image = Image.open(image_file).convert("RGB")
image = np.array(image)


def inference_time_tpu(edgetpu_model_name):
    """Calculate the inference time for a model running on the Edge TPU.

    Parameters:
    ----------------
    edgetpu_model_name : str
        The filename of the model that is being executed on the Edge TPU.

    Returns:
    ----------------
    tpu_inference_time : float
        The inference time in milliseconds.
    """
    # Create an interpreter object for the Edge TPU model
    interpreter = make_interpreter(edgetpu_model_name)

    # Allocate memory for tensors in the interpreter
    interpreter.allocate_tensors()

    # Get the details of the input tensor for the model
    input_details = interpreter.get_input_details()[0]

    # Prepare the input image tensor and cast it to the appropriate data type
    input_tensor = np.expand_dims(image, axis=0).astype(input_details["dtype"])

    # Set the input tensor for the interpreter
    interpreter.set_tensor(input_details["index"], input_tensor)

    # Record the start time, run the interpreter, and calculate the inference time
    start_time = time.monotonic()
    interpreter.invoke()
    tpu_inference_time = (time.monotonic() - start_time) * 1000

    return tpu_inference_time
