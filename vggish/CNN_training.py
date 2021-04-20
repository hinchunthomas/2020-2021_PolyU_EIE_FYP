from __future__ import absolute_import, division, print_function
import os
import sys
from tensorflow import compat as tf
import numpy as np
import tf_slim as slim
import vggish_params
import utils
import sklearn.model_selection

# os.makedirs("/models/research/audioset/vggish/audio_classifier_CNN")
sys.path.append(os.path.join('.', '..'))
flags = tf.v1.flags
tf = tf.v1

flags.DEFINE_string(
    'tfrecord_file_name', 'Evalval2_training',
    'Path to a TFRecord file where embeddings will be written.')

flags.DEFINE_integer(
    'num_batches', 128,
    'Number of batches of examples to feed into the model. Each batch is of '
    'variable size and contains shuffled examples of each class of audio.')

flags.DEFINE_integer(
    'training_epochs', 10,
    'Number of training loops for the model'
)

FLAGS = flags.FLAGS

def main(_):
    """
        Traning my own CNN model with VGGish
        => Finished the evaluation of the base model
        To Do:
            => Increase the number of fully-connected layer (Done)
            => Prediction function (Done)
            => Optimizer function (Done)
            => Loss function (Done)
    """
    # Read embeddings amd their labels
    (embedding_train, embedding_labels_train) = utils.read_TFRecord_File(FLAGS.tfrecord_file_name)
    embedding_train = np.array(embedding_train)
    embedding_labels_train = np.array(embedding_labels_train)
    num_features = embedding_train[0].shape[0]
    num_classes = embedding_labels_train[0].shape[0]
    print('Number of features:', num_features)
    
    # Separate the training embedding and labels into two set (Training and Validation)
    examples_train, examples_val, labels_train, labels_val = sklearn.model_selection.train_test_split(
        embedding_train, 
        embedding_labels_train, 
        test_size=0.33
    )

    # Control the GPU resources without usih all the resources
    gpu_opts = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)

    with tf.Graph().as_default(), tf.Session(config=tf.ConfigProto(gpu_options=gpu_opts)) as sess:

        with tf.variable_scope('Label'):
            labels_input = tf.placeholder(
                tf.float32, shape=(None, num_classes), name='labels')

        with tf.variable_scope('Feature'):
            features_input = tf.placeholder(
                tf.float32, shape=(None, num_features), name='features')

        with tf.variable_scope('Model'):
            num_units_1 = 100
            num_units_2 = 50

            fc_1 = slim.fully_connected(tf.nn.relu(features_input), num_units_1)
            fc_2 = slim.fully_connected(fc_1, num_units_2)

            # Add classifier layer(s) at the end
            logits = slim.fully_connected(
                fc_2, num_classes, activation_fn=None, scope='logits')
            prediction = tf.math.softmax(logits, name='prediction')

        # Add training ops.
        with tf.variable_scope('Train'):
            global_step = tf.train.create_global_step()

        with tf.variable_scope('Cross-entropy'):
            # Cross-entropy label loss.
            xent = tf.nn.softmax_cross_entropy_with_logits(
                logits=logits, labels=labels_input, name='xent')
            loss = tf.reduce_mean(xent, name='loss_op')
            tf.summary.scalar('loss', loss)

        with tf.variable_scope('Optimizer'):
            # We use the same optimizer and hyperparameters as used to train VGGish.
            optimizer = tf.train.AdamOptimizer(
                learning_rate=vggish_params.LEARNING_RATE,
            )
            train_op = optimizer.minimize(loss, global_step=global_step)

        with tf.variable_scope('Accuracy'):
            # Accuracy of the model
            # Correct or not
            correct_prediction = tf.equal(
                tf.argmax(prediction, 1),
                tf.argmax(labels_input, 1)
            )
            acc = tf.reduce_mean(
                tf.cast(correct_prediction, tf.float32),
                name='acc_op'
            )
            tf.summary.scalar('Accuracy', acc)

        sess.run(tf.global_variables_initializer())
        gpu_opts = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)

        for te in range(FLAGS.training_epochs):
            print('Training loop: ', te + 1)
            for _ in range(FLAGS.num_batches):
                # Shuffle the example
                (train_features, train_labels) = utils.shuffle_embedding(examples_train, labels_train)
                (val_features, val_labels) = utils.shuffle_embedding(examples_val, labels_val)

                # Training Step
                [num_steps, train_loss_value, train_accuracy, _] = sess.run(
                        [global_step, loss, acc, train_op],
                        feed_dict={features_input: train_features, labels_input: train_labels})

                # Validation Step
                [num_steps, val_loss_value, val_accuracy, _] = sess.run(
                        [global_step, loss, acc, train_op],
                        feed_dict={features_input: val_features, labels_input: val_labels})

                # Print the loss and accuracy when it comes to the last batch
                # if (_ == 127):
                print(te + 1, ': ', num_steps + 1, 'Train Loss: ', train_loss_value)
                print(te + 1, ': ', num_steps + 1, 'Train Accuracy: ', train_accuracy)
                print('-------')
                print(te + 1, ': ', num_steps + 1, 'Validation Loss', val_loss_value)
                print(te + 1, ': ', num_steps + 1, 'Validation Accuracy', val_accuracy)
                print('-------')
        
        saver = tf.train.Saver()
        saver.save(sess, "audio_classifier_CNN/audio_classifier_CNN_softmax_cross_entropy_with_logits_Evalval2")

if __name__ == '__main__':
    tf.app.run()