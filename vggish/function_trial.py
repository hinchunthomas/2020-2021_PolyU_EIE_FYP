from __future__ import print_function

from tensorflow import compat as tf
import numpy as np

import utils

tf = tf.v1

def main(_):
    class_test = 0
    class_train = 0
    (features_train, features_test, labels_train, labels_test) = utils.collect_example_and_label()
    
    for i in range(0, len(labels_test)):
        label = np.array(labels_test[i])
        class_number = label.argmax()
        if (class_number == 6):
            class_test += 1

    for i in range(0, len(labels_train)):
        label = np.array(labels_train[i])
        class_number = label.argmax()
        if (class_number == 6):
            class_train += 1

    print('labels_test', class_test)
    print('labels_train', class_train)

if __name__ == '__main__':
    tf.app.run()