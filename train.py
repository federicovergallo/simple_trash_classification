#!/usr/bin/env python
# coding: utf-8
import tensorflow as tf
import tensorflow.keras.layers as kl
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from dataset_preprocessing import DataPreprocessing
import datetime
import os
from tqdm import tqdm

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

BATCH_SIZE = 16
LR = 1e-05
EPOCHS = 300
use_pretrained = True
data_augmentation = True
pretrained_name = 'mobilenet'
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
checkpoint_dir = 'checkpoints/'
log_dir = 'logs/'

if use_pretrained:
    checkpoint_dir += pretrained_name
    log_dir += pretrained_name
else:
    checkpoint_dir += 'scratch'
    log_dir += 'scratch'

if data_augmentation:
    checkpoint_dir += '_aug_lite'
    log_dir += '_aug_lite'

train_summary_writer = tf.summary.create_file_writer(log_dir)


def create_log_dir(log_dir, checkpoint_dir):
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    if not os.path.exists('checkpoints'):
        os.mkdir('checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)


def network():
    model = tf.keras.Sequential()
    model.add(kl.InputLayer(input_shape=(224, 224, 3)))
    if use_pretrained:
        if pretrained_name == 'vgg':
            vgg = VGG16(weights='imagenet', input_shape=(224, 224, 3), include_top=False)
            vgg.trainable = False
            model.add(vgg)
        if pretrained_name == 'mobilenet':
            mobnet = MobileNetV2(weights='imagenet', input_shape=(224, 224, 3), include_top=False)
            mobnet.trainable = False
            model.add(mobnet)
    else:
        # First conv block
        model.add(kl.Conv2D(filters=128, kernel_size=3, padding='same', strides=2))
        model.add(tf.keras.layers.ReLU())
        model.add(kl.MaxPooling2D(pool_size=(2, 2)))
        # Second conv block
        model.add(kl.Conv2D(filters=256, kernel_size=3, padding='same', strides=2))
        model.add(tf.keras.layers.ReLU())
        model.add(kl.MaxPooling2D(pool_size=(2, 2)))
        # Third conv block
        model.add(kl.Conv2D(filters=512, kernel_size=3, padding='same', strides=2))
        model.add(tf.keras.layers.ReLU())
        model.add(kl.MaxPooling2D(pool_size=(2, 2)))
    # Flatten
    model.add(kl.Flatten())
    # First FC
    model.add(kl.Dense(256, activation="relu"))
    # Dropout
    model.add(kl.Dropout(0.2))
    # Second Fc
    model.add(kl.Dense(128, activation="relu"))
    # Dropout
    model.add(kl.Dropout(0.2))
    # Output FC with sigmoid at the end
    model.add(kl.Dense(3, activation='softmax', name='prediction'))
    return model
'''
https://keras.io/guides/writing_a_training_loop_from_scratch/
Compile into a static graph any function that take tensors as input to apply global performance optimizations.
'''

@tf.function
def macro_f1(y, y_hat, thresh=0.5):
    """Compute the macro F1-score on a batch of observations (average F1 across labels)

    Args:
        y (int32 Tensor): labels array of shape (BATCH_SIZE, N_LABELS)
        y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)
        thresh: probability value above which we predict positive

    Returns:
        macro_f1 (scalar Tensor): value of macro F1 for the batch
    """
    y_pred = tf.cast(tf.greater(y_hat, thresh), tf.float32)
    tp = tf.cast(tf.math.count_nonzero(y_pred * y, axis=0), tf.float32)
    fp = tf.cast(tf.math.count_nonzero(y_pred * (1 - y), axis=0), tf.float32)
    fn = tf.cast(tf.math.count_nonzero((1 - y_pred) * y, axis=0), tf.float32)
    f1 = 2 * tp / (2 * tp + fn + fp + 1e-16)
    macro_f1 = tf.reduce_mean(f1)
    return macro_f1

@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model(x)
        loss_value = loss_fn(y, logits)
    grads = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    # Track progress
    train_loss_avg(loss_value)
    #train_accuracy.update_state(macro_f1(y, logits))
    train_accuracy(y, logits)
    return

@tf.function
def test_step(model, x, y, set_name):
    logits = model(x)
    if set_name == 'val':
        val_accuracy(y, logits)
    else:
        test_accuracy(y, logits)


def split_dataset(dataset: tf.data.Dataset, validation_data_fraction: float):
    """
    Splits a dataset of type tf.data.Dataset into a training and validation dataset using given ratio. Fractions are
    rounded up to two decimal places.
    @param dataset: the input dataset to split.
    @param validation_data_fraction: the fraction of the validation data as a float between 0 and 1.
    @return: a tuple of two tf.data.Datasets as (training, validation)
    """

    validation_data_percent = round(validation_data_fraction * 100)
    if not (0 <= validation_data_percent <= 100):
        raise ValueError("validation data fraction must be ∈ [0,1]")

    dataset = dataset.enumerate()
    train_dataset = dataset.filter(lambda f, data: f % 100 > validation_data_percent)
    validation_dataset = dataset.filter(lambda f, data: f % 100 <= validation_data_percent)

    # remove enumeration
    train_dataset = train_dataset.map(lambda f, data: data)
    validation_dataset = validation_dataset.map(lambda f, data: data)

    return train_dataset, validation_dataset


if __name__ == '__main__':
    preprocessing_class = DataPreprocessing()

    # Open each dataset
    # --- Cans datasets ---
    cans_tfrecord = 'dataset/tfrecords/cans.tfrecords'
    cans_dataset = tf.data.TFRecordDataset(cans_tfrecord)
    # Splitting in train and test set
    cans_train_set, cans_test_set = split_dataset(cans_dataset, 0.75)


    # --- metal datasets ---
    metal_tfrecord = 'dataset/tfrecords/metal.tfrecords'
    metal_dataset = tf.data.TFRecordDataset(metal_tfrecord)
    # Splitting in train and test set
    metal_train_set, metal_test_set = split_dataset(metal_dataset, 0.75)

    # --- Oranges datasets ---
    oranges_tfrecord = 'dataset/tfrecords/oranges.tfrecords'
    oranges_dataset = tf.data.TFRecordDataset(oranges_tfrecord)
    # Splitting in train and test set
    oranges_train_set, oranges_test_set = split_dataset(oranges_dataset, 0.75)

    # --- Plastic datasets ---
    plastic_tfrecord = 'dataset/tfrecords/plastic.tfrecords'
    plastic_dataset = tf.data.TFRecordDataset(plastic_tfrecord)
    # Splitting in train and test set
    plastic_train_set, plastic_test_set = split_dataset(plastic_dataset, 0.75)

    # --- plastic_web datasets ---
    plastic_web_tfrecord = 'dataset/tfrecords/plastic_web.tfrecords'
    plastic_web_dataset = tf.data.TFRecordDataset(plastic_web_tfrecord)
    # Splitting in train and test set
    plastic_web_train_set, plastic_web_test_set = split_dataset(plastic_web_dataset, 0.75)

    # Merging them
    # -- Training set --
    train_set = cans_train_set.concatenate(oranges_train_set)
    train_set = train_set.concatenate(plastic_train_set)
    train_set = train_set.concatenate(metal_train_set)
    train_set = train_set.concatenate(plastic_web_train_set)
    # -- Test set --
    test_set = cans_test_set.concatenate(oranges_test_set)
    test_set = test_set.concatenate(plastic_test_set)
    test_set = test_set.concatenate(metal_test_set)
    test_set = test_set.concatenate(plastic_web_test_set)

    # Parse the record into tensors with map.
    train_set = train_set.map(preprocessing_class.decode)
    train_set = train_set.shuffle(1)
    train_set = train_set.batch(BATCH_SIZE)

    # Parse the record into tensors with map.
    test_set = test_set.map(preprocessing_class.decode)
    test_set = test_set.shuffle(1)
    test_set = test_set.batch(BATCH_SIZE)

    # Create the model
    model = network()
    print(model.summary())

    # Optimizers and metrics
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    accuracy_fn = tf.keras.metrics.Accuracy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=LR)

    train_loss_avg = tf.keras.metrics.Mean()
    train_accuracy = tf.keras.metrics.CategoricalAccuracy()
    test_accuracy = tf.keras.metrics.CategoricalAccuracy()
    val_accuracy = tf.keras.metrics.CategoricalAccuracy()

    create_log_dir(log_dir, checkpoint_dir)

    last_models = sorted(os.listdir(checkpoint_dir))
    if last_models:
        last_model_path = checkpoint_dir + '/' + last_models[-1]
        first_epoch = int(last_models[-1].split("_")[1]) + 1
        print("First epoch is ", first_epoch)
        model = tf.keras.models.load_model(last_model_path)
    else:
        first_epoch = 0
        model = network()

    # Train
    for epoch in tqdm(range(first_epoch, EPOCHS+1), total=EPOCHS+1-first_epoch):
        # Reset the metrics at the start of the next epoch
        train_loss_avg.reset_states()
        train_accuracy.reset_states()
        val_accuracy.reset_states()
        test_accuracy.reset_states()
        try:
            # Training loop
            for x_batch_train, y_batch_train in train_set:
                # Do step
                train_step(x_batch_train, y_batch_train)

            # Write in the summary
            with train_summary_writer.as_default():
                tf.summary.scalar('Train Loss', train_loss_avg.result(), step=epoch)
                tf.summary.scalar('Train Accuracy', train_accuracy.result(), step=epoch)
                # Test on validation set
                if epoch % 25 == 0:
                    for x_batch_test, y_batch_test in test_set:
                        test_step(model, x_batch_test, y_batch_test, 'test')
                    tf.summary.scalar('Test Accuracy', test_accuracy.result(), step=epoch)

            if epoch % 25 == 0:
                tf.keras.models.save_model(model, '{}/Epoch_{}_model.hp5'.format(checkpoint_dir, str(epoch)),
                                           save_format="h5")

        except KeyboardInterrupt:
            print("Keyboard Interruption...")
            # Save model
            tf.keras.models.save_model(model, '{}/Epoch_{}_model.hp5'.format(checkpoint_dir, str(epoch)),
                                       save_format="h5")
            break

    # Test on validation set
    for x_batch_test, y_batch_test in test_set:
        test_step(model, x_batch_test, y_batch_test, 'test')
    test_set_acc = test_accuracy.result().numpy()
    print("Accuracy on test set is", test_set_acc)