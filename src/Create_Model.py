from keras import layers
import tensorflow_addons as tfa
from tensorflow import keras
import tensorflow as tf

from src.Decode_Block import decoded_block
from src.Gene_Pool import conv_block


def get_spatial_dimensions(layer):
    """
    Retrieve the spatial dimensions of a layer's output.

    Parameters:
    ----------------
    layer : keras Layer
        A layer in a Keras model.

    Returns:
    ----------------
    tuple or None
        A tuple of spatial dimensions if they exist, or None if the layer has no spatial dimensions.
    """
    output_shape = layer.output_shape
    if isinstance(output_shape, list):
        output_shape = output_shape[0]  # Use the first item in the list

    if len(output_shape) == 4:  # Check if the layer has spatial dimensions
        return output_shape[1:3]  # Return only the spatial

    return None  # Return None if no spatial dimensions


def find_layers_with_downsampling(model):
    """
    Find all layers in a Keras model that reduce the spatial dimensions of their input.

    Parameters:
    ----------------
    model : keras Model
        A Keras model.

    Returns:
    ----------------
    layers_with_downsampling : list
        A list of Keras layers that reduce the spatial dimensions of their input.
    """
    layers_with_downsampling = []

    for i in range(len(model.layers) - 1):
        current_layer = model.layers[i]
        next_layer = model.layers[i + 1]

        current_shape = get_spatial_dimensions(current_layer)
        next_shape = get_spatial_dimensions(next_layer)

        if current_shape is None or next_shape is None:  # Skip layers without spatial dimensions
            continue

        if current_shape[0] > next_shape[0] and current_shape[1] > next_shape[1]:
            if "rescaling" not in current_layer.name:
                layers_with_downsampling.append(current_layer)

    return layers_with_downsampling


def create_base_model(model_array, input_shape=(160, 160, 3)):
    """
    Create the base model used in the main U-Net model.

    Parameters:
    ----------------
    model_array : np.array
        A numpy array representing the architecture of the model.

    input_shape : tuple, default=(160, 160, 3)
        The shape of the input to the model.

    Returns:
    ----------------
    model : keras Model
        A Keras model representing the base model.
    """
    inputs = layers.Input(shape=input_shape)
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    x = conv_block(x, kernel_size=2, filters=64, strides=2)

    for i in range(9):
        x = decoded_block(x, model_array[i])

    model = keras.Model(inputs, x)

    return model


def create_model(model_array, input_shape=(160, 160, 3), num_classes=34):
    """
    Create a U-Net model based on the given architecture array.

    Parameters:
    ----------------
    model_array : np.array
        A numpy array representing the architecture of the model.

    input_shape : tuple, default=(160, 160, 3)
        The shape of the input to the model.

    num_classes : int, default=34
        The number of classes the model should predict.

    Returns:
    ----------------
    unet : keras Model
        A Keras model representing the U-Net.
    """
    base_model = create_base_model(model_array, input_shape)

    # Find the layers with downsampling in the base model
    downsampling_layers = find_layers_with_downsampling(base_model)

    # Get output of the downsampling layers for skip connections
    skip_outputs = [layer.output for layer in downsampling_layers]

    # Encoder
    encoder_output = base_model.output

    # Decoder
    x = encoder_output
    for i in range(len(skip_outputs) - 1, -1, -1):
        skip_output = skip_outputs[i]
        x = tf.keras.layers.UpSampling2D(size=2, interpolation='bilinear')(x)
        x = tf.keras.layers.Conv2D(skip_output.shape[-1], kernel_size=3, padding='same', activation=tf.nn.relu)(x)
        x = tf.image.resize(x, skip_output.shape[1:3])
        x = tf.keras.layers.Concatenate()([x, skip_output])
        x = tf.keras.layers.Conv2D(filters=skip_output.shape[-1], kernel_size=3, padding='same', activation=tf.nn.relu)(
            x)

    # Final upsampling layer and output
    x = tf.keras.layers.UpSampling2D(size=2, interpolation='bilinear')(x)
    x = tf.keras.layers.Conv2D(filters=num_classes, kernel_size=3, padding='same', activation=tf.nn.relu)(x)
    output = tf.keras.layers.Conv2D(num_classes, kernel_size=1, activation=tf.nn.softmax)(x)
    output = tf.image.resize(output, base_model.input.shape[1:3])

    # Create and compile the U-Net model
    unet = tf.keras.Model(inputs=base_model.input, outputs=output)

    return unet


def model_summary(model):
    """
    Print the summary of a Keras model and the number of its trainable weights.

    Parameters:
    ----------------
    model : keras Model
        A Keras model.
    """
    model.summary()
    print('Number of trainable weights = {}'.format(len(model.trainable_weights)))


def train_model(train_ds, val_ds,
                model, epochs=30,
                checkpoint_filepath="checkpoints/checkpoint"):
    """
    Train a Keras model on the provided datasets.

    Parameters:
    ----------------
    train_ds : tf.data.Dataset
        A tf.data.Dataset object for the training data.

    val_ds : tf.data.Dataset
        A tf.data.Dataset object for the validation data.

    model : keras Model
        The Keras model to train.

    epochs : int, default=30
        The number of epochs to train for.

    checkpoint_filepath : str, default="checkpoints/checkpoint"
        The file path where the model checkpoint will be saved.

    Returns:
    ----------------
    model : keras Model
        The trained Keras model.

    history : keras History
        A History object. Its History.history attribute is a record of training loss values and metrics values at
        successive epochs, as well as validation loss values and validation metrics values (if applicable).
    """
    checkpoint_callback = keras.callbacks.ModelCheckpoint(checkpoint_filepath,
                                                          monitor="val_accuracy",
                                                          save_best_only=True,
                                                          save_weights_only=True)

    loss_fn = keras.losses.CategoricalCrossentropy()

    opt = tfa.optimizers.LazyAdam(learning_rate=0.004)
    opt = tfa.optimizers.MovingAverage(opt)
    opt = tfa.optimizers.Lookahead(opt)

    metrics = ["accuracy", tf.keras.metrics.MeanIoU(num_classes=34, name='mean_io_u')]

    model.compile(optimizer=opt,
                  loss=loss_fn,
                  metrics=metrics)

    try:
        history = model.fit(train_ds,
                            epochs=epochs,
                            validation_data=val_ds,
                            callbacks=[checkpoint_callback])

        model.load_weights(checkpoint_filepath)
    except Exception as e:
        history = None
        print(e)
    return model, history
