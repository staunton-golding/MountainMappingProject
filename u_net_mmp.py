from typing import Union, Any

import numpy as np
import cv2
from glob import glob
import matplotlib.pyplot as plt

import os
import shutil
import random

from numpy import ndarray, dtype, floating
from numpy._typing import _32Bit

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Dropout, Lambda, ReLU, BatchNormalization, Layer
from tensorflow.keras.utils import to_categorical
from IPython.display import clear_output
from tensorflow_examples.models.pix2pix import pix2pix

#For running TF on mac (no GPU)
TF_CPP_MIN_LOG_LEVEL = 0
tf.config.set_visible_devices([], 'GPU')

#split data into training, testing, and validation splits
def split_dataset_train_test_val(input_dir: str, output_dir: str, split_ratio: tuple[int,int,int] = (0.7, 0.15, 0.15)) -> None:

    train_dir_image = os.path.join(output_dir, "train", "images")
    train_dir_mask = os.path.join(output_dir, "train", "labels")

    test_dir_image = os.path.join(output_dir, "test", "images")
    test_dir_mask = os.path.join(output_dir, "test", "labels")

    val_dir_image = os.path.join(output_dir, "val", "images")
    val_dir_mask = os.path.join(output_dir, "val", "labels")

    # Create output image directories if they don't exist
    os.makedirs(train_dir_image, exist_ok=True)
    os.makedirs(test_dir_image, exist_ok=True)
    os.makedirs(val_dir_image, exist_ok=True)

    # Create output mask directories if they don't exist
    os.makedirs(train_dir_mask, exist_ok=True)
    os.makedirs(test_dir_mask, exist_ok=True)
    os.makedirs(val_dir_mask, exist_ok=True)

    #input for image and mask
    input_dir_image = os.path.join(input_dir, "images")
    input_dir_mask = os.path.join(input_dir, "labels")

    # Get a list of all image files (assuming they are .jpg)
    image_files = [f for f in os.listdir(input_dir_image) if f.endswith('.jpg') or f.endswith('.png')]
    random.shuffle(image_files)

    # Calculate split point
    split_point_1 = int(len(image_files) * split_ratio[0])
    split_point_2 = int(len(image_files) * (split_ratio[0] + split_ratio[1]))

    # Split into train and test sets
    train_files = image_files[:split_point_1]
    test_files = image_files[split_point_1:split_point_2]
    val_files = image_files[split_point_2:]

    print(f"Total files: {len(image_files)}")
    print(f"Training files: {len(train_files)}")
    print(f"Testing files: {len(test_files)}")
    print(f"Validation Files: {len(val_files)}")

    # Copy files to respective directories
    x = 0
    for filename in train_files:
        # Assuming masks have a similar name but different extension or prefix
        #[item.partition('.')[0]
        mask_filename = filename.partition('_')[0] + "_mask.jpg"
        shutil.copy(os.path.join(input_dir_image, filename), os.path.join(train_dir_image, filename))
        if os.path.exists(os.path.join(input_dir_mask, mask_filename)):
            shutil.copy(os.path.join(input_dir_mask, mask_filename), os.path.join(train_dir_mask, mask_filename))
            x = x + 1
        else:
            print(f"Warning: Mask not found for {filename}")

    for filename in test_files:
        mask_filename = filename.partition('_')[0] + "_mask.jpg"
        shutil.copy(os.path.join(input_dir_image, filename), os.path.join(test_dir_image, filename))
        if os.path.exists(os.path.join(input_dir_mask, mask_filename)):
            shutil.copy(os.path.join(input_dir_mask, mask_filename), os.path.join(test_dir_mask, mask_filename))
        else:
            print(f"Warning: Mask not found for {filename}")

    for filename in val_files:
        mask_filename = filename.partition('_')[0] + "_mask.jpg"
        shutil.copy(os.path.join(input_dir_image, filename), os.path.join(val_dir_image, filename))
        if os.path.exists(os.path.join(input_dir_mask, mask_filename)):
            shutil.copy(os.path.join(input_dir_mask, mask_filename), os.path.join(val_dir_mask, mask_filename))
        else:
            print(f"Warning: Mask not found for {filename}")

#split data into training, and testing splits
def split_dataset_train_test(input_dir: str, output_dir: str, split_ratio: tuple[int,int] = (0.8, 0.2)):

    train_dir_image = os.path.join(output_dir, "train", "images")
    train_dir_mask = os.path.join(output_dir, "train", "labels")

    test_dir_image = os.path.join(output_dir, "test", "images")
    test_dir_mask = os.path.join(output_dir, "test", "labels")

    # Create output image directories if they don't exist
    os.makedirs(train_dir_image, exist_ok=True)
    os.makedirs(test_dir_image, exist_ok=True)

    # Create output mask directories if they don't exist
    os.makedirs(train_dir_mask, exist_ok=True)
    os.makedirs(test_dir_mask, exist_ok=True)

    # input for image and mask
    input_dir_image = os.path.join(input_dir, "images")
    input_dir_mask = os.path.join(input_dir, "labels")

    # Get a list of all image files (assuming they are .jpg)
    image_files = [f for f in os.listdir(input_dir_image) if f.endswith('.jpg') or f.endswith('.png')]
    random.shuffle(image_files)

    # Calculate split point
    split_point = int(len(image_files) * split_ratio[0])

    # Split into train and test sets
    train_files = image_files[:split_point]
    test_files = image_files[split_point:]

    print(f"Total files: {len(image_files)}")
    print(f"Training files: {len(train_files)}")
    print(f"Testing files: {len(test_files)}")

    # Copy files to respective directories
    x = 0
    for filename in train_files:
        # Assuming masks have a similar name but different extension or prefix
        # [item.partition('.')[0]
        mask_filename = filename.partition('_')[0] + "_mask.jpg"
        shutil.copy(os.path.join(input_dir_image, filename), os.path.join(train_dir_image, filename))
        if os.path.exists(os.path.join(input_dir_mask, mask_filename)):
            shutil.copy(os.path.join(input_dir_mask, mask_filename), os.path.join(train_dir_mask, mask_filename))
            x = x + 1
        else:
            print(f"Warning: Mask not found for {filename}")

    for filename in test_files:
        mask_filename = filename.partition('_')[0] + "_mask.jpg"

        shutil.copy(os.path.join(input_dir_image, filename), os.path.join(test_dir_image, filename))
        if os.path.exists(os.path.join(input_dir_mask, mask_filename)):
            shutil.copy(os.path.join(input_dir_mask, mask_filename), os.path.join(test_dir_mask, mask_filename))
        else:
            print(f"Warning: Mask not found for {filename}")

#can change input data path if data is already split
input_data_path = "./training_data_resized"
output_split_path = "./train_test_data_split"
already_split = True

if already_split:
    print(f"data already split, can be found at {output_split_path}")
else:
    split_dataset_train_test(input_data_path, output_split_path)

#paramters involving the loading of data
IMG_SIZE = 256
BATCH_SIZE = 20
AUTOTUNE = tf.data.AUTOTUNE
NUM_CHANNELS = 3

#load data, read image and mask, preprocess data, create tf_dataset
def load_data(path: str) -> tuple[list[str], list[str]]:
    images = sorted(glob(os.path.join(path, "images/*.jpg")))
    masks = sorted(glob(os.path.join(path, "labels/*.jpg")))
    return images, masks


def read_image(path: str) -> ndarray[Any, dtype[floating[_32Bit]]]:
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (IMG_SIZE, IMG_SIZE))
    x = x / 255.0
    x = x.astype(np.float32)
    return x


def read_mask(path: str) -> ndarray[Any, dtype[floating[_32Bit]]]:
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    x = x / 255.0
    x[x > 1] = 1
    x[x < 1] = 0
    x = x.astype(np.float32)
    return x


def preprocess(x: object, y: object) -> tuple[Any, Any]:
    def f(x1: bytes, y1: bytes) -> tuple[ndarray[Any, dtype[floating[_32Bit]]], ndarray[Any, dtype[floating[_32Bit]]]]:
        x1 = x1.decode()
        y1 = y1.decode()

        x1 = read_image(x1)
        y1 = read_mask(y1)
        return x1, y1

    images, masks = tf.numpy_function(f, [x, y], [tf.float32, tf.float32])
    images.set_shape([IMG_SIZE, IMG_SIZE, NUM_CHANNELS])
    masks.set_shape([IMG_SIZE, IMG_SIZE])
    masks = tf.expand_dims(masks, axis=-1)
    return images, masks


def tf_dataset(x: object, y: object) -> Any:
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.map(preprocess)
    dataset = dataset.prefetch(2)
    return dataset

train_path = f"{output_split_path}/train/"
train_images, train_masks = load_data(train_path)
train_ds = tf_dataset(train_images, train_masks)

test_path = f"{output_split_path}/test/"
test_images, test_masks = load_data(test_path)
test_ds = tf_dataset(test_images, test_masks)

#small dataset, not using val data
"""
val_path = f"{output_split_path}/val/"
val_images, val_masks = load_data(val_path)
val_ds = tf_dataset(val_images, val_masks)
"""

#augment data
def augment(image_mask: object, num_channels: int = NUM_CHANNELS) -> tuple[Any, Any]:
    image, mask = image_mask

    # Make seeds for each augmentation
    initial_seed = [42, 0]

    #Random crop back to the original size.
    image_to_crop = tf.image.resize_with_crop_or_pad(image, IMG_SIZE + 6, IMG_SIZE + 6)
    mask_to_crop = tf.image.resize_with_crop_or_pad(mask, IMG_SIZE + 6, IMG_SIZE + 6)

    image = tf.image.stateless_random_crop(image_to_crop, size=[image_to_crop.shape[0]-6, image_to_crop.shape[1]-6, image_to_crop.shape[2]], seed=initial_seed)
    mask = tf.image.stateless_random_crop(mask_to_crop, size=[IMG_SIZE, IMG_SIZE, 1], seed=initial_seed)

    #Random change brightness
    image = tf.image.stateless_random_brightness(image, max_delta=0.5, seed=initial_seed)

    #Random change contrast
    image = tf.image.stateless_random_contrast(image, lower=0.1, upper=0.9, seed=initial_seed)

    #Random flip images L-R
    image = tf.image.stateless_random_flip_left_right(image, seed=initial_seed)
    mask = tf.image.stateless_random_flip_left_right(mask, seed=initial_seed)

    #Random change jpeg quality
    image = tf.image.stateless_random_jpeg_quality(image, min_jpeg_quality=85, max_jpeg_quality=95, seed=initial_seed)

    if num_channels == 3:
        #Random change saturation of image
        image = tf.image.stateless_random_saturation(image, lower=0.1, upper=1.0, seed=initial_seed)

        # Random change hue
        image = tf.image.stateless_random_hue(image, max_delta=0.5, seed=initial_seed)

    #Clip all adjusted images to values between 0 and 1
    image = tf.clip_by_value(image, 0, 1)
    return image, mask


counter = tf.data.experimental.Counter()
train_ds_counter = tf.data.Dataset.zip((train_ds, (counter, counter)))

train_ds = (
    train_ds_counter
    .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .cache()
    .shuffle(len(train_images))
    .repeat()
    .prefetch(AUTOTUNE)
)

test_ds = (
    test_ds
    .repeat()
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

#This is the U-NET based on the semantic segmentation tutorial from TensorFlow
TRAIN_LENGTH = len(train_images)
#og unet
base_model = tf.keras.applications.MobileNetV2(input_shape=[IMG_SIZE, IMG_SIZE, NUM_CHANNELS], include_top=False)

# Use the activations of these layers
layer_names = [
    'block_1_expand_relu',  # 64x64
    'block_3_expand_relu',  # 32x32
    'block_6_expand_relu',  # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',  # 4x4
]
base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

# Create the feature extraction model
down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)

down_stack.trainable = False

up_stack = [
    pix2pix.upsample(512, NUM_CHANNELS),  # 4x4 -> 8x8
    pix2pix.upsample(256, NUM_CHANNELS),  # 8x8 -> 16x16
    pix2pix.upsample(128, NUM_CHANNELS),  # 16x16 -> 32x32
    pix2pix.upsample(64, NUM_CHANNELS),  # 32x32 -> 64x64
]

def unet_model_tf_tutorial(output_channels: int) -> Any:
    inputs = tf.keras.layers.Input(shape=[IMG_SIZE, IMG_SIZE, NUM_CHANNELS])

    # Downsampling through the model
    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    # This is the last layer of the model
    last = tf.keras.layers.Conv2DTranspose(filters=output_channels, kernel_size=3, strides=2,
                                           padding='same')  #64x64 -> 128x128

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


#U-NET Model based on literature (not on tutorial).
#Canny edge detection in skip connections to preserve edge-information (ideally, with dice, will weigh only pertinent canny) (spoiler, it did not really work).
#for applying canny to skip connections
def apply_canny(image_tensor: object) -> Any:
    # Convert TensorFlow tensor to NumPy array
    image_np = image_tensor.numpy()

    if np.max(np.max(np.max(
            image_np))) < 1.1:  #if image is normalized (should be, converts back to original scale for canny)
        image_np = image_np * 255  #convert to full scale
    image_np = image_np.astype('uint8')  #convert to uint8

    # Ensure the image is grayscale if not already
    if len(image_np.shape) == 3 and image_np.shape[2] == 3:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    edges_full = []

    #canny across channels means up to 512 canny images per layer
    for jj in range(image_np.shape[0]):
        edges_temp = []
        for ii in range(image_np.shape[-1]):
            #derive thresholds for canny from otsu
            high_thresh, thresh_im = cv2.threshold(image_np[jj, :, :, ii], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            low_thresh = 0.33 * high_thresh
            edges_temp.append(cv2.Canny(image_np[jj, :, :, ii], threshold1=low_thresh,
                                        threshold2=high_thresh))  # Adjust thresholds as needed
        edges_temp = np.stack(edges_temp)
        edges_temp = np.moveaxis(edges_temp, 0, -1)
        edges_full.append(edges_temp)
    edges = np.array(edges_full)

    # Convert the result back to a TensorFlow tensor
    edges = edges / 255
    edges_tensor = tf.convert_to_tensor(edges, dtype=tf.uint8)
    return edges_tensor


#turn canny into tf function
def preprocess_image_with_canny(image_tensor: object) -> Any:
    # tf.py_function requires specifying the output types
    canny_output = tf.py_function(func=apply_canny,
                                  inp=[image_tensor],
                                  Tout=tf.uint8)
    # Ensure the shape is defined for static analysis
    canny_output.set_shape(image_tensor.shape)
    return canny_output


#canny tf_fn wrapped in a layer
class CannyAppLayer(Layer):
    def call(self, x):
        return preprocess_image_with_canny(x)


# Building the convolutional block
def conv_block(inputs: object, filters:int =64) -> Any:
    # Taking first input and implementing the first conv block
    conv1 = Conv2D(filters, kernel_size=(3, 3), padding="same")(inputs)
    batch_norm1 = BatchNormalization()(conv1)
    activ1 = ReLU()(batch_norm1)

    # Taking first input and implementing the second conv block
    conv2 = Conv2D(filters, kernel_size=(3, 3), padding="same")(activ1)
    batch_norm2 = BatchNormalization()(conv2)
    activ2 = ReLU()(batch_norm2)

    return activ2

#Encoder (down sample) blocks
def encoder_block(filters: int, inputs: object, dropout_rate: float) -> tuple[Any, Any]:
    s = conv_block(inputs, filters)
    p = MaxPooling2D(pool_size=(2, 2), padding='same')(s)
    p = Dropout(dropout_rate)(p)
    return s, p  #p provides the input to the next encoder block and s provides the context/features to the symmetrically opposite decoder block

#Baseline layer is just a bunch on Convolutional Layers to extract high level features from the downsampled Image (bottleneck)
def baseline_layer(filters: int, inputs: object) -> Any:
    x = conv_block(inputs, filters)
    return x

#Decoder Block (with canny in skip connections)
def decoder_block_canny(filters: int, connections: object, inputs: object, dropout_rate: float) -> Any:
    up_samp = Conv2DTranspose(filters, kernel_size=(2, 2), padding='same', activation='relu', strides=2)(inputs)
    canny_ds = CannyAppLayer()(up_samp)
    skip_connections = tf.keras.layers.Concatenate(axis=-1)([up_samp, connections, canny_ds])
    x = conv_block(skip_connections, filters)
    x = Dropout(dropout_rate)(x)
    return x

#decoder block (old school, no canny)
def decoder_block_non_canny(filters: int, connections: object, inputs: object, dropout_rate: float) -> Any:
    up_samp = Conv2DTranspose(filters, kernel_size=(3, 3), padding='same', activation='relu', strides=2)(inputs)
    skip_connections = tf.keras.layers.Concatenate(axis=-1)([up_samp, connections])
    x = conv_block(skip_connections, filters)
    x = Dropout(dropout_rate)(x)
    return x

def unet(d1_can: bool, d2_can: bool, d3_can: bool, d4_can: bool, dr_r1: float, dr_r2: float, dr_r3: float, dr_r4: float) -> Any:
    #Defining the input layer and specifying the shape of the images
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, NUM_CHANNELS))
    print(f"input expected shape is {inputs.shape}")

    #defining the encoder
    s1, p1 = encoder_block(64, inputs=inputs, dropout_rate=dr_r1)
    s2, p2 = encoder_block(128, inputs=p1, dropout_rate=dr_r2)
    s3, p3 = encoder_block(256, inputs=p2, dropout_rate=dr_r3)
    s4, p4 = encoder_block(512, inputs=p3, dropout_rate=dr_r4)

    #Setting up the baseline
    baseline = baseline_layer(1024, p4)

    #Defining the entire decoder
    if d1_can:
        d1 = decoder_block_canny(512, s4, baseline, dropout_rate=dr_r_4)
    else:
        d1 = decoder_block_non_canny(512, s4, baseline, dropout_rate=dr_r_4)

    if d2_can:
        d2 = decoder_block_canny(256, s3, d1, dropout_rate=dr_r_3)
    else:
        d2 = decoder_block_non_canny(256, s3, d1, dropout_rate=dr_r_3)

    if d3_can:
        d3 = decoder_block_canny(128, s2, d2, dropout_rate=dr_r_2)
    else:
        d3 = decoder_block_non_canny(128, s2, d2, dropout_rate=dr_r_2)

    if d4_can:
        d4 = decoder_block_canny(64, s1, d3, dropout_rate=dr_r_1)
    else:
        d4 = decoder_block_non_canny(64, s1, d3, dropout_rate=dr_r_1)

    #Setting up the output function for binary classification of pixels
    outputs = Conv2D(OUTPUT_CLASSES, kernel_size=(1,1), activation='sigmoid')(d4)

    #Finalizing the model
    model = Model(inputs=inputs, outputs=outputs, name='Unet')

    return model

#for metrics, dice coefficient
def dice_coeff(y_true: list[Union[float,int]], y_pred: list[Union[float,int]], smooth: int = 1) -> Union[float,int]:
    intersection = tf.reduce_sum(y_true * y_pred, axis=-1)
    union = tf.reduce_sum(y_true, axis=-1) + tf.reduce_sum(y_pred, axis=-1)
    dice_coeff_return = (2 * intersection + smooth) / (union + smooth)
    return dice_coeff_return


#NOTE: I know keras has a Dice function. I just wanted to see how I can implement my own loss functions
class CustomDiceLoss(tf.keras.losses.Loss):
    def __init__(self, name='custom_dice_loss'):
        super().__init__(name=name)

    def call(self, y_true: list[Union[float,int]], y_pred: list[Union[float,int]], smooth: int = 1) -> Union[float,int]:
        intersection = tf.reduce_sum(y_true * y_pred, axis=-1)
        union = tf.reduce_sum(y_true, axis=-1) + tf.reduce_sum(y_pred, axis=-1)
        dice_coeff_return = (2 * intersection + smooth) / (union + smooth)
        return 1 - dice_coeff_return

    # Required for serialization: allows saving and loading the model
    def get_config(self):
        config = super().get_config()
        config.update({'threshold': self.threshold})
        return config


custom_dice_loss_instance = CustomDiceLoss()

#callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr_unet = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                   patience=5, min_lr=1e-7)

#u-net model selection parameters
tutorial = False
#canny in skip connections
d1_can = False
d2_can = False
d3_can = False
d4_can = False
#dropout rates
dr_r_1 = .0625
dr_r_2 = .125
dr_r_3 = .25
dr_r_4 = .35
OUTPUT_CLASSES = 2

if tutorial:
    model_unet = unet_model_tf_tutorial(output_channels=OUTPUT_CLASSES)
else:
    model_unet = unet(d1_can, d2_can, d3_can, d4_can, dr_r_1, dr_r_2, dr_r_3, dr_r_4)

model_unet.summary()

lr = 0.00005
optimizer_use = tf.keras.optimizers.Adam(learning_rate=lr)
model_unet.compile(optimizer=optimizer_use,
                   loss=custom_dice_loss_instance,
                   metrics=['accuracy', dice_coeff])

EPOCHS = 300
VALIDATION_STEPS = 10

BUFFER_SIZE = 1000
STEPS_PER_EPOCH = 9#TRAIN_LENGTH // BATCH_SIZE
early_stopping_bin = True

if tutorial:
    out_dir_gif = (
    f"./{EPOCHS}_epochs_{lr}_learning_rate_{STEPS_PER_EPOCH}_steps_per_epoch_{BATCH_SIZE}_"
    f"batch_size_{TRAIN_LENGTH}_train_length_{BUFFER_SIZE}_buff_size_{IMG_SIZE}_img_size_TUTORIAL_DICE_early_stopping_reduce_LR")
    
else:
    out_dir_gif = (
    f"./{EPOCHS}_epochs_{lr}_learning_rate_{STEPS_PER_EPOCH}_steps_per_epoch_{BATCH_SIZE}_"
    f"batch_size_{TRAIN_LENGTH}_train_length_{BUFFER_SIZE}_buff_size_{IMG_SIZE}_img_size_{d1_can}_1c_{d2_can}_2c_{d3_can}_3c_{d4_can}_4c_{dr_r_1}_dr1_"
    f"{dr_r_2}_dr2_{dr_r_3}_dr3_{dr_r_4}_dr4_DICE_early_stopping_sigmoid_reduce_LR_with_BN")

#NOTE, WHEN CHANGING LOSS FUNCTION, CHANGE SCE TO ABBREVIATION FOR LOSS FUNCTION
#ALSO, early stopping and reduce LR pretty much always a good idea. If you want to change that, make sure to also change end of out_dir_gif name

os.makedirs(out_dir_gif, exist_ok=True)


def display(display_list: object, epoch: object) -> None:
    plt.figure(figsize=(15, 15))

    #first image is Input Image, second is True Mask, third is Predicted Mask
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
        plt.axis('off')

    plt.savefig(f"{out_dir_gif}/predicted mask after {epoch} epochs")
    plt.close()


def create_mask(pred_mask: ndarray) -> Any:
    pred_mask = tf.math.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]


sample_image_mask = test_ds.take(1)
sample_image = list(sample_image_mask)[0][0][0]
sample_mask = list(sample_image_mask)[0][1][0]


def show_predictions(dataset: object = None, num: int = 1, epoch_use: int = 0) -> None:
    if dataset:
        for image, mask in dataset.take(num):
            pred_mask = model_unet.predict(image)
            display([image[0], mask[0], create_mask(pred_mask)], epoch = epoch_use)
    else:
        display([sample_image, sample_mask, create_mask(model_unet.predict(sample_image[tf.newaxis, :]))],
                epoch=epoch_use)


class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch: int, logs: object = None) -> Any:
        clear_output(wait=True)
        show_predictions(epoch_use = epoch)
        print ('\nSample Prediction after epoch {}\n'.format(epoch+1))


model_history = model_unet.fit(train_ds, epochs=EPOCHS,
                               steps_per_epoch=STEPS_PER_EPOCH,
                               validation_steps=VALIDATION_STEPS,
                               validation_data=test_ds,
                               callbacks=[DisplayCallback(), early_stopping, reduce_lr_unet])

show_predictions()
loss = model_history.history['loss']
val_loss = model_history.history['val_loss']

plt.figure()
plt.plot(model_history.epoch, loss, 'r', label='Training loss')
plt.plot(model_history.epoch, val_loss, 'bo', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.ylim([0, 1])
plt.legend()
plt.savefig(f"{out_dir_gif}/training_and_validation_loss")

loss, accuracy, dice = model_unet.evaluate(test_ds, verbose=1)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")
print(f"Test DICE: {dice}")

