from __future__ import print_function

from random import shuffle

import os
import shutil
import numpy as np
import tensorflow as tf_2
from tensorflow import compat as tf

import tf_slim as slim
import sklearn.model_selection

import vggish_input

tf = tf.v1

def collect_example_and_label():
    """Returns a shuffled batch of examples of all audio classes.

    Note that this is just a toy function because this is a simple demo intended
    to illustrate how the training code might work.

    Returns:
    a tuple (features, labels) where features is a NumPy array of shape
    [batch_size, num_frames, num_bands] where the batch_size is variable and
    each row is a log mel spectrogram patch of shape [num_frames, num_bands]
    suitable for feeding VGGish, while labels is a NumPy array of shape
    [batch_size, num_classes] where each row is a multi-hot label vector that
    provides the labels for corresponding rows in features.
    """
    # Folder directory for each class
    car_horn_directory = #r'location_of_the_sound_samples_folder'
    children_playing_directory = #r'location_of_the_sound_samples_folder'
    dog_bark_directory = #r'location_of_the_sound_samples_folder'
    gun_shot_directory = #r'location_of_the_sound_samples_folder'
    siren_directory = #r'location_of_the_sound_samples_folder'
    bell_directory = #r'location_of_the_sound_samples_folder'
    construction_sound_directory = #r'location_of_the_sound_samples_folder'

    # Make Example list for each class
    all_examples = np.zeros((1, 96, 64))
    (all_examples, car_horn) = make_example(car_horn_directory, all_examples)
    (all_examples, children_playing) = make_example(children_playing_directory, all_examples)
    (all_examples, dog_bark) = make_example(dog_bark_directory, all_examples)
    # (all_examples, drilling) = make_example(drilling_directory, all_examples)
    (all_examples, gun_shot) = make_example(gun_shot_directory, all_examples)
    # (all_examples, jackhammer) = make_example(jackhammer_directory, all_examples)
    (all_examples, siren) = make_example(siren_directory, all_examples)
    (all_examples, bell) = make_example(bell_directory, all_examples)
    (all_examples, construction_sound) = make_example(construction_sound_directory, all_examples)
    all_examples = np.delete(all_examples, 0, axis=0)

    # Label list for each list
    all_labels = np.array([[1, 0, 0, 0, 0, 0, 0]] * car_horn[0].shape[0])

    # Make labels for each example in th list for each class
    for example in car_horn:
        example_array = np.array([[1, 0, 0, 0, 0, 0, 0]] * example.shape[0])
        all_labels = np.concatenate((all_labels, example_array))

    
    for example in children_playing:
        example_array = np.array([[0, 1, 0, 0, 0, 0, 0]] * example.shape[0])
        all_labels = np.concatenate((all_labels, example_array))

    for example in dog_bark:
        example_array = np.array([[0, 0, 1, 0, 0, 0, 0]] * example.shape[0])
        all_labels = np.concatenate((all_labels, example_array))

    for example in gun_shot:
        example_array = np.array([[0, 0, 0, 1, 0, 0, 0]] * example.shape[0])
        all_labels = np.concatenate((all_labels, example_array))

    for example in siren:
        example_array = np.array([[0, 0, 0, 0, 1, 0, 0]] * example.shape[0])
        all_labels = np.concatenate((all_labels, example_array))

    for example in bell:
        example_array = np.array([[0, 0, 0, 0, 0, 1, 0]] * example.shape[0])
        all_labels = np.concatenate((all_labels, example_array))

    for example in construction_sound:
        example_array = np.array([[0, 0, 0, 0, 0, 0, 1]] * example.shape[0])
        all_labels = np.concatenate((all_labels, example_array))

    for i in range(car_horn[0].shape[0]):
        all_labels = np.delete(all_labels, 0, axis=0)

    examples_train, examples_test, labels_train, labels_test = sklearn.model_selection.train_test_split(all_examples, all_labels)
    # print('labels_test', labels_test)
    return (examples_train, examples_test, labels_train, labels_test)

def make_example(class_directory, all_examples):
    """
        Arg:
            class_directory: The directory of the class
            all_examples: The array for all example

        Return:
            Examples (Spectrogram) of each wavfile in the list
    """

    file_name_list = []
    examples_list = []

    for file_name in os.listdir(class_directory):
        file_name_list.append(os.path.join(class_directory, file_name))

    for file_name in file_name_list:
        if(vggish_input.wavfile_to_examples(file_name).shape[0] != 0):
            all_examples = np.concatenate((all_examples, vggish_input.wavfile_to_examples(file_name)))
            examples_list.append(vggish_input.wavfile_to_examples(file_name))

    return (all_examples, examples_list)

def waveform_to_examples_and_label(file_name):

    file_name = os.path.join(#r'location_of_the_sound_samples_folder'
        , file_name)
    # Convert the waveform to the example
    waveform_examples = vggish_input.wavfile_to_examples(file_name)

    # Label the example (this is for car horn => [1, 0, 0, 0, 0, 0, 0, 0, 0])
    labels = np.array([[0, 0, 0, 1, 0, 0, 0]] * waveform_examples.shape[0])

    return (waveform_examples, labels)


def _bytes_feature(value):
    return tf_2.train.Feature(bytes_list=tf_2.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf_2.train.Feature(int64_list=tf_2.train.Int64List(value=value))

def read_TFRecord_File(file_path):
    tfrecords_filename = 'tfRecords/'+ file_path +'.tfrecords'
    record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)
    embedding_tot = np.zeros((1, 128))
    embedding_labels_tot = []
    embedding_labels = []

    for string_record in record_iterator:

        example = tf_2.train.Example()
        example.ParseFromString(string_record)
        embedding_labels = []

        for i in range(7):
            value = int(example.features.feature[file_path + '/labels'].int64_list.value[i])
            embedding_labels.append(value)

        embedding_raw = (example.features.feature[file_path + '/embedding'].bytes_list.value[0])

        embedding_1d = np.fromstring(embedding_raw, dtype=np.float32)
        #print(embedding_1d.shape)
        embedding_labels_tot.append(embedding_labels)
        #print(embedding_1d[0:10])
        reconstructed_embedding = embedding_1d.reshape((-1, 128))
        embedding_tot = np.append(embedding_tot, reconstructed_embedding, axis=0)

    embedding_tot = np.delete(embedding_tot, 0, axis=0)
    print('embedding shape: ', embedding_tot.shape)
    print('number of embedding: ', len(embedding_labels_tot))

    return(embedding_tot, embedding_labels_tot)

def shuffle_embedding(features, labels):
    # Combine all examples and labels from each class
    labeled_examples = list(zip(features, labels))
    shuffle(labeled_examples)

    # Separate and return the features and labels.
    features = [example for (example, _) in labeled_examples]
    labels = [label for (_, label) in labeled_examples]
    return (features, labels)
