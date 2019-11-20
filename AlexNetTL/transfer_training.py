#
import numpy as np
from cifar import Cifar
import pretrained
from tqdm import tqdm
import tensorflow as tf
import helper


def train():
    n_classes = 10
    learning_rate = 0.00001
    batch_size = 16
    num_opochs = 100
    num_test_splits = 10
    image_size = 224

    conv5 = tf.layers.flatten(pretrained.maxpool5)

    weights = tf.Variable(tf.zeros([9216, n_classes]), name='output_weight')
    bias = tf.Variable(tf.truncated_normal([n_classes]), name='output_bias')
    model = tf.matmul(conv5, weights) + bias

    outputs = tf.placeholder(tf.float32, [None, n_classes])

    cost = tf.losses.softmax_cross_entropy(outputs, model)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    correct_pred = tf.equal(tf.argmax(model,1), tf.argmax(outputs, 1))
    accuracy = tf.reduce_sum(tf.cast(correct_pred, tf.float32))

    cifar = Cifar(batch_size=batch_size)
    cifar.create_resized_test_set(dim=n_classes)

    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(num_opochs):
            for i in tqdm(range(len(cifar.batches)), desc="Epoch {}".format(epoch), unit="batch"):
                this_batch = cifar.batch(i)
                input_batch, out = helper.reshape_batch(this_batch, image_size), n_classes

                sess.run([optimizer], feed_dict={pretrained.x: input_batch, outputs: out})

            inp_test, out_test = cifar.test_set
            inp_test = np.split(inp_test, num_test_splits)
            out_test = np.split(out_test, num_test_splits)

            total_acc = 0
            for each_inp_test, each_out_test in tqdm(zip(inp_test, out_test),
                                                     desc='Test'.format(epoch), unit=' batch ', total=num_test_splits):
                each_test_acc = sess.run(accuracy, feed_dict={pretrained.x: each_inp_test, outputs: each_out_test})
                total_acc = total_acc + each_test_acc

            test_acc = total_acc/num_test_splits
            print('Test ACC: {}'.format(test_acc))


def main(_):
    train()


if __name__ == '__main__':
    tf.app.run(main=main)
