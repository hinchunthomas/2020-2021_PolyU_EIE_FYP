from __future__ import print_function

from random import shuffle

import os
import numpy as np
from tensorflow import compat as tf
import tf_slim as slim
import sklearn.model_selection

import utils
import vggish_input
import vggish_params
import vggish_slim
import shutil

# tf.enable_eager_execution()

flags = tf.v1.flags
tf = tf.v1

flags.DEFINE_integer(
    'num_batches', 50,
    'Number of batches of examples to feed into the model. Each batch is of '
    'variable size and contains shuffled examples of each class of audio.')

flags.DEFINE_integer(
    'training_epochs', 15,
    'NUmber of training loops for the model'
)

flags.DEFINE_boolean(
    'train_vggish', False,
    'If True, allow VGGish parameters to change during training, thus '
    'fine-tuning VGGish. If False, VGGish parameters are fixed, thus using '
    'VGGish as a fixed feature extractor.')

flags.DEFINE_string(
    'checkpoint', 'vggish_model.ckpt',
    'Path to the VGGish checkpoint file.')

flags.DEFINE_string(
    'new_checkpoint_sig_1', 'fyp_vggish_model_sig_1.ckpt',
    'Path to the FYP VGGish checkpoint file (1st trial)'
)

FLAGS = flags.FLAGS

_NUM_CLASSES = 9

def _get_examples_batch(example_train, label_train):
    # Combine all examples and labels from each class
    labeled_examples = list(zip(example_train, label_train))
    shuffle(labeled_examples)

    # Separate and return the features and labels.
    features = [example for (example, _) in labeled_examples]
    labels = [label for (_, label) in labeled_examples]
    return (features, labels)

def main(_):
    """
        Traning the VGG-ish model with my own classification layer
    """
    with tf.Graph().as_default(), tf.Session() as sess:
        # Define VGGish.
        embeddings = vggish_slim.define_vggish_slim(training=FLAGS.train_vggish)
        print(embeddings)
        # Define a shallow classification model and associated training ops on top of VGGish.
        with tf.variable_scope('Model'):
            # Add a fully connected layer with 100 units. Add an activation function
            # to the embeddings since they are pre-activation.
            num_units = 100
            fc = slim.fully_connected(tf.nn.relu(embeddings), num_units)

            # Add a classifier layer at the end, consisting of parallel logistic
            # classifiers, one per class. This allows for multi-class tasks.
            logits = slim.fully_connected(
                fc, _NUM_CLASSES, activation_fn=None, scope='logits')
            prediction = tf.sigmoid(logits, name='prediction')

            # Next Trial
            # prediction = tf.round(tf.nn.sigmoid(logit, name='prediction'))
            # And delete the checkpoint file and download again
            # Also separate into training set and testing set (See if there is any library for separation) (done)
            # Then add training epochs (done), batch size (done) (See the previous Google Colab Notebook)
            # See this can bring back to Google Colab train

        # Add training ops.
        with tf.variable_scope('Train'):
            global_step = tf.train.create_global_step()

        with tf.variable_scope('Label'):
            # Labels are assumed to be fed as a batch multi-hot vectors, with
            # a 1 in the position of each positive class label, and 0 elsewhere.
            labels_input = tf.placeholder(
                tf.float32, shape=(None, _NUM_CLASSES), name='labels')

        with tf.variable_scope('Cross-entropy'):
            # Cross-entropy label loss.
            xent = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=logits, labels=labels_input, name='xent')
            loss = tf.reduce_mean(xent, name='loss_op')
            tf.summary.scalar('loss', loss)

        with tf.variable_scope('Optimizer'):
            # We use the same optimizer and hyperparameters as used to train VGGish.
            optimizer = tf.train.AdamOptimizer(
                learning_rate=vggish_params.LEARNING_RATE,
                epsilon=vggish_params.ADAM_EPSILON)
            train_op = optimizer.minimize(loss, global_step=global_step)

        with tf.variable_scope('Accuracy'):
            # Accuracy of the model
            # Correct or not
            correct_prediction = tf.equal(
                tf.argmax(prediction, 1), 
                tf.argmax(labels_input,1)
            )
            acc = tf.reduce_mean(
                tf.cast(correct_prediction, tf.float32),
                name='acc_op'
            )
            tf.summary.scalar('Accuracy', acc)

        # Initialize all variables in the model, and then load the pre-trained
        # VGGish checkpoint.
        sess.run(tf.global_variables_initializer())
        vggish_slim.load_vggish_slim_checkpoint(sess, FLAGS.checkpoint)

        # merged = tf.summary.merge_all()
        # writer = tf.summary.FileWriter(logs_path, graph = tf.get_default_graph())

        # The training loop.
        features_input = sess.graph.get_tensor_by_name(
            vggish_params.INPUT_TENSOR_NAME)

        (features_train, features_test, labels_train, labels_test) = utils.collect_example_and_label()
        
        for te in range(FLAGS.training_epochs):
            print('Training loop: ', te + 1)
            for _ in range(FLAGS.num_batches):
                (features, labels) = _get_examples_batch(features_train, labels_train)
                [num_steps, loss_value, _] = sess.run(
                        [global_step, loss, train_op],
                        feed_dict={features_input: features, labels_input: labels})
                print('Step %d: loss %g' % (num_steps + 1, loss_value))
                # print('Step %d: accuracy %g' % (num_steps + 1, acc_value))
                print('-------')

        # Saving the new model with new checkpoint file
        # print('\nSaving')
        # cwd = os.getcwd()
        # path = os.path.join(cwd, 'simple')
        # shutil.rmtree(path, ignore_errors=True)
        # inputs_dict = {
        #     "features_input": features_input, 
        #     "labels_input": labels_input
        # }
        # outputs_dict = {
        #         "logits": logits
        #     }
        # tf.saved_model.simple_save(
        #         sess, path, inputs_dict, outputs_dict
        #     )
        # print('Ok')

        # Next Trial
        print('Label for validation')
        print(sess.run(
            labels_input,
            feed_dict={features_input: features_test, labels_input: labels_test}
        ))
        print("---")
        print("Testing Accuracy: ", sess.run(acc, feed_dict={features_input: features_test, labels_input: labels_test}))
        print("------")

        # Validation with one class of sound event
        # (feature_validation, label_validation) = 

if __name__ == '__main__':
    tf.app.run()