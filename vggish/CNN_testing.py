from __future__ import absolute_import, division, print_function
import os
import sys
from tensorflow import compat as tf
import numpy as np
import tf_slim as slim
import vggish_params
import utils

flags = tf.v1.flags
tf = tf.v1

flags.DEFINE_string(
    'tfrecord_file_name', 'gun_shot_testing_after_trimming',
    'Path to a TFRecord file where embeddings will be written.')

FLAGS = flags.FLAGS

def main(_):
    # If you want to test new single waveform file with different classes
    # Here are some steps for you before you run this python script

    # 1. Go to "feature_extracter_and_recorder.py"
    #   => Change the "tfrecord_file"/file path/feature name

    # 2. Go to "utils.py"
    #   => Change the labels array in the function "waveform_to_examples_and_label(file_name)"

    # 3. Go to "CNN_testing.py"
    #   => Change the "tfrecord_file_name"

    # 4. Run "feature_extracter_and_recorder.py"

    # 5. Put the tfrecord file into the folder "tfRecords"

    # 6. Run "feature_extracter_and_recorder.py" and "CNN_testing.py"

    (embedding_test, embedding_labels_test) = utils.read_TFRecord_File(FLAGS.tfrecord_file_name)

    gpu_opts = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
    # load the trained network from a local drive
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_opts)) as sess:
        #First let's load meta graph and restore weights
        saver = tf.train.import_meta_graph("audio_classifier_CNN/audio_classifier_CNN_softmax_cross_entropy_with_logits_Evalval2.meta")
        saver.restore(sess, tf.train.latest_checkpoint('audio_classifier_CNN/'))
        
        # Access and create placeholders variables and
        # create feed-dict to feed new data
        graph = tf.get_default_graph()
        # all_names = [op.name for op in graph.get_operations()]
        # print(all_names)
        features_input = graph.get_tensor_by_name("Feature/features:0")
        feed_dict = {features_input: embedding_test}

        # Access the op that you want to run.
        prediction = graph.get_tensor_by_name("Model/prediction:0")

        y_pred = sess.run(prediction, feed_dict)

        pred = sess.run(tf.argmax(y_pred, axis=1))
        #print("class predicion embedding 1:", pred)
        #print("real label: ",embedding_labels_test[0:100])

    correct_pred = 0
    wrong_pred = 0

    for i in range(0, len(pred)):
        label = np.array(embedding_labels_test[i])
        class_number = label.argmax()
        if pred[i] == class_number:
            correct_pred += 1
        else:
            wrong_pred += 1
            print("Wrong embedding number: ", i + 1)
            print('Prediction class: ', pred[i])
            print('Actual class: ', class_number)
            print('-------')

    acc = correct_pred/len(pred)
    print('Total number of prediction:', len(pred))
    print('Number of wrong prediction: ', wrong_pred)
    print("Test accuracy: ", acc)

if __name__ == '__main__':
    tf.app.run()
