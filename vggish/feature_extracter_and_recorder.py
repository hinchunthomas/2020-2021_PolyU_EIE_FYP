from __future__ import print_function

import sys

from tensorflow import compat as tf

import vggish_params
import vggish_slim
import utils

flags = tf.v1.flags
tf = tf.v1

flags.DEFINE_string(
    'checkpoint', 'vggish_model.ckpt',
    'Path to the VGGish checkpoint file.')

flags.DEFINE_string(
    'tfrecord_file', 'training.tfrecords',
    'Path to a TFRecord file where embeddings will be written.')

FLAGS = flags.FLAGS

def main(_):
    # Extract the feature amd labels from sound library
    (features_train, features_test, labels_train, labels_test) = utils.collect_example_and_label()

    num_of_feature_train = 0
    num_of_feature_test = 0

    for i in range(len(features_train)):
        num_of_feature_train += 1
    
    for i in range(len(features_test)):
        num_of_feature_test += 1

    print('Number of training feature', num_of_feature_train)
    print('Number of testing feature', num_of_feature_test)

    # Extract the feature amd labels from a single file
    # (single_file_features, single_file_labels) = utils.waveform_to_examples_and_label('gun_shot_testing_sound.wav')

    writer = tf.python_io.TFRecordWriter(FLAGS.tfrecord_file)

    # restricting memory usage, TensorFlow is greedy and will use all memory otherwise
    config = tf.ConfigProto()
    config.gpu_options.allocator_type = 'BFC'
    config.gpu_options.per_process_gpu_memory_fraction = 0.90

    with tf.Graph().as_default(), tf.Session(config=config) as sess:
        # Define the model in inference mode, load the checkpoint, and
        # locate input and output tensors.
        vggish_slim.define_vggish_slim(training=False)
        vggish_slim.load_vggish_slim_checkpoint(sess, FLAGS.checkpoint)
        features_tensor = sess.graph.get_tensor_by_name(
            vggish_params.INPUT_TENSOR_NAME)
        embedding_tensor = sess.graph.get_tensor_by_name(
            vggish_params.OUTPUT_TENSOR_NAME)

        # Run inference
        [embedding_batch] = sess.run([embedding_tensor], feed_dict={features_tensor: features_train})

        # print('example_batch shape: ', features_test.shape)
        # print('embedding_batch shape: ', len(embedding_batch))
        # print('labels_batch shape: ', len(labels_test))
    
    for i in range(len(embedding_batch)):
        embedding = embedding_batch[i]

        # convert into proper data type:
        embedding_label = labels_train[i] # embedding.shape[0]
        embedding_raw = embedding.tobytes()

        if(i == 1):
            print('label', embedding_label)

        # Create a feature
        feature = {'training/labels': utils._int64_feature(embedding_label),
                'training/embedding':  utils._bytes_feature(embedding_raw)}

        if(i == 1):
            print('label', feature)

        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        if(i == 1):
            print('example', example)
        # Serialize to string and write on the file

        writer.write(example.SerializeToString())

    writer.close()
    sys.stdout.flush()

if __name__ == '__main__':
    tf.app.run()