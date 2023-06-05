import os
import glob
import cv2
import concurrent.futures
import numpy as np
import tensorflow as tf
import random

# Specify batch size and optimization level
batch_size = 24
auto = tf.data.AUTOTUNE
# Specify the number of target classes
num_classes = 34


# Function to get directories of images and annotations
def get_image_and_annotation_dirs(data_split):
    # Construct directories for the images and annotations
    x_dir = os.path.join("cityscapes", "leftImg8bit_trainvaltest", "leftImg8bit", data_split)
    y_dir = os.path.join("cityscapes", "gtFine_trainvaltest", "gtFine", data_split)

    # Get list of image and annotation files, sort and pair them
    images_dir, annotations_dir = zip(*sorted([(x, y.replace("gtFine_color", "gtFine_labelIds"))
                                               for x, y in zip(glob.glob(os.path.join(x_dir, "*", "*.png")),
                                                               glob.glob(
                                                                   os.path.join(y_dir, "*", "*gtFine_color.png")))]))

    # Pair and shuffle image and annotation directories
    combined = list(zip(images_dir, annotations_dir))
    random.shuffle(combined)
    images_dir_shuffled, annotations_dir_shuffled = zip(*combined)

    return images_dir_shuffled, annotations_dir_shuffled


# Function to read an image file and resize it,
# with an optional parameter to specify if the image is an annotation
def read_and_resize_image(img_path, size, is_annotation=False):
    flags = cv2.IMREAD_UNCHANGED if is_annotation else cv2.IMREAD_COLOR
    img = cv2.imread(img_path, flags)
    resized_img = cv2.resize(img, size)
    return tf.keras.utils.to_categorical(resized_img, num_classes=34) if is_annotation else resized_img


# Function to process a list of image paths in parallel
def read_and_resize_images(image_paths, size, is_annotation=False):
    def process_image(img_path):
        return read_and_resize_image(img_path, size, is_annotation)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        images = list(executor.map(process_image, image_paths))
    return np.array(images)


# Get training and validation image and annotation directories
x_train_dir, y_train_dir = get_image_and_annotation_dirs('train')
x_val_dir, y_val_dir = get_image_and_annotation_dirs('val')

size = (160, 160)
# Load and preprocess the training, validation, and test images and annotations
x_train = read_and_resize_images(x_train_dir, size)[:1024]
y_train = read_and_resize_images(y_train_dir, size, is_annotation=True)[:1024]
x_val = read_and_resize_images(x_val_dir, size)[:128]
y_val = read_and_resize_images(y_val_dir, size, is_annotation=True)[:128]
x_test = read_and_resize_images(x_train_dir, size)[1300:1428]
y_test = read_and_resize_images(y_train_dir, size, is_annotation=True)[1300:1428]

# Construct TensorFlow data pipeline for training, validation, and test data
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size).prefetch(auto)
val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size).prefetch(auto)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size).prefetch(auto)
