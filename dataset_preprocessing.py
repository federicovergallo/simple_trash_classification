#!/usr/bin/env python3
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import cv2
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import random


AUTOTUNE = tf.data.experimental.AUTOTUNE
batch_size = 32
np.random.seed(0)


class DataPreprocessing:
    def __init__(self,
                 IMG_HEIGHT=224,
                 IMG_WIDTH=224,
                 dataset_dir='dataset/',
                 ):
        self.IMG_HEIGHT = IMG_HEIGHT
        self.IMG_WIDTH = IMG_WIDTH
        self.dataset_dir = dataset_dir


    def resize(self, image):
        return cv2.resize(image, (self.IMG_HEIGHT, self.IMG_WIDTH), interpolation=cv2.INTER_AREA)


    def random_crop(self, image, crop_height, crop_width):
        max_x = image.shape[1] - crop_width
        max_y = image.shape[0] - crop_height

        x = np.random.randint(0, max_x)
        y = np.random.randint(0, max_y)

        crop = image[y: y + crop_height, x: x + crop_width]

        return crop


    def augment_image(self, image):
        '''
        Applies some augmentation techniques
        '''
        # Mirror flip
        flipped = tf.image.flip_left_right(image).numpy()
        # Transpose flip
        transposed = tf.image.transpose(image).numpy()
        # Gaussian Blur
        gaussian = cv2.GaussianBlur(image, (5, 5), cv2.BORDER_DEFAULT)
        # Brightness
        brightness = tf.image.adjust_brightness(image, 0.4).numpy()
        # Contrast
        contrast = tf.image.random_contrast(image, lower=0.0, upper=1.0).numpy()
        # Resize at the end
        images = [self.resize(image) for image in [flipped, transposed, gaussian, brightness, contrast]]
        return images

    def _int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    def _bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def writeTfRecord(self, output_dir, data_augmentation=False):
        '''
        Method to write tfrecord
        '''
        # Datasets name and label
        sets = [('metal', '0'),
                ('cans', '0'),
                ('oranges', '1'),
                ('plastic_web', '2'),
                ('plastic', '2')]

        # open the TFRecords file
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for setname, label in tqdm(sets):
            # Open Writer
            writer = tf.io.TFRecordWriter(output_dir+setname+'.tfrecords')
            set_folder = self.dataset_dir+setname
            # Get all the images of a set
            images_path = [set_folder+"/"+img for img in os.listdir(set_folder)]
            for image_path in tqdm(images_path, total=len(images_path)):
                # Read the image from path
                img = cv2.imread(image_path)[..., ::-1]
                # Create a feature
                if data_augmentation:
                    images = self.augment_image(img)
                else:
                    images = [img]
                for image in images:
                    feature = {'image': self._bytes_feature(tf.compat.as_bytes(image.tostring())),
                               'label': self._bytes_feature(tf.compat.as_bytes(label))}
                    # Create an example protocol buffer
                    example = tf.train.Example(features=tf.train.Features(feature=feature))

                    # Serialize to string and write on the file
                    writer.write(example.SerializeToString())
            writer.close()

    def decode(self, serialized_example):
        """
        Parses an image and label from the given `serialized_example`.
        It is used as a map function for `dataset.map`
        """
        IMAGE_SHAPE = (self.IMG_HEIGHT, self.IMG_WIDTH, 3)

        # 1. define a parser
        features = tf.io.parse_single_example(
            serialized_example,
            # Defaults are not specified since both keys are required.
            features={
                'image': tf.io.FixedLenFeature([], tf.string),
                'label': tf.io.FixedLenFeature([], tf.string),
            })

        # 2. Convert the data
        image = tf.io.decode_raw(features['image'], tf.uint8)
        label = tf.strings.to_number(features['label'])
        #label = tf.io.decode_raw(features['label'], tf.uint8)

        # Cast
        label = tf.cast(label, tf.int32)
        label = tf.squeeze(tf.one_hot(label, depth=3))

        # 3. reshape
        image = tf.convert_to_tensor(tf.reshape(image, IMAGE_SHAPE))

        return image, label


if __name__ == '__main__':
    preprocessing_class = DataPreprocessing()
    # Write tf recordfloat32
    preprocessing_class.writeTfRecord('dataset/tfrecords/', data_augmentation=True)

    # Read TfRecord
    tfrecord_path = 'dataset/tfrecords/metal.tfrecords'
    dataset = tf.data.TFRecordDataset(tfrecord_path)

    # Parse the record into tensors with map.
    # map takes a Python function and applies it to every sample.
    dataset = dataset.map(preprocessing_class.decode)

    # Divide in batch
    dataset = dataset.batch(batch_size)

    # Create an iterator
    iterator = iter(dataset)
    a = iterator.get_next()

    # Element of iterator
    plt.imshow(iterator.get_next()[0][0].numpy())
    plt.show()
    '''
    # Join together tfrecords files
    dir1 = 'dataset/tfrecords/cans.tfrecords'
    dir2 = 'dataset/tfrecords/oranges.tfrecords'
    dir3 = 'dataset/tfrecords/plastic.tfrecords'
    # Create dataset from multiple .tfrecord files
    list_of_tfrecord_files = [dir1, dir2, dir3]
    dataset = tf.data.TFRecordDataset(list_of_tfrecord_files)

    # Save dataset to .tfrecord file
    filename = 'dataset/tfrecords/dataset.tfrecord'
    writer = tf.data.experimental.TFRecordWriter(filename)
    writer.write(dataset)
    '''


