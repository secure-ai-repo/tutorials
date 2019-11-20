"""
For the purpose of upgrading the development of Tensorflow AI Vision Project
Edited by Dr. Hyeung-yun Kim
"""

import numpy as np
import argparse
import os
import sys
import tensorflow as tf
import matplotlib.pyplot as plt

from progressbar import ProgressBar
from tensorflow.examples.tutorials.mnist import input_data


def train():
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    def conv2d(input, weight_shape, bias_shape):
        weight_prod = weight_shape[0] * weight_shape[1] * weight_shape[2]
        weight_init = tf.random_normal_initializer(stddev=(2.0/weight_prod)**0.5)
        bias_init = tf.constant_initializer(value=0)
        W = tf.get_variable('W', weight_shape, initializer=weight_init)
        b = tf.get_variable('b', bias_shape, initializer=bias_init)
        con_out = tf.nn.conv2d(input, W, strides=[1,1,1,1], padding='SAME')
        return tf.nn.relu(tf.nn.bias_add(con_out, b))

    def max_pool(in_pool, k=2):
        return tf.nn.max_pool(in_pool, ksize=[1,k,k,1], strides=[1,k,k,1], padding='SAME')

    def layer(input_layer, weight_shape, bias_shape):
        weight_init = tf.random_normal_initializer(stddev=(2.0/weight_shape[0])**0.5)
        bias_init = tf.constant_initializer(value=0)
        W = tf.get_variable('W', weight_shape, initializer=weight_init)
        b = tf.get_variable('b', bias_shape, initializer=bias_init)
        return tf.nn.relu(tf.matmul(input_layer, W) + b)

    def inference(x, keep_prob):
        x = tf.reshape(x, shape=[-1, 28, 28, 1])

        with tf.variable_scope('conv_1'):
            conv_1 = conv2d(x, [5, 5, 1, 32], [32])
            pool_1 = max_pool(conv_1)

        with tf.variable_scope('conv_2'):
            conv_2 = conv2d(pool_1, [5, 5, 32, 64], [64])
            pool_2 = max_pool(conv_2)

        with tf.variable_scope('fc'):
            pool_2_flat = tf.reshape(pool_2, [-1, 7*7*64])
            fc_1 = layer(pool_2_flat, [7*7*64, 1024], [1024])
            fc_1_drop = tf.nn.dropout(fc_1, keep_prob=keep_prob)

        with tf.variable_scope('output'):
            output = layer(fc_1_drop, [1024, 10], [10])
        return output

    def loss(out_loss, y_loss):
        xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=out_loss, labels=y_loss)
        loss_op = tf.reduce_mean(xentropy)
        return loss_op

    def training(loss_step, gl_step):

        # 1. Run this optimizer
        tf.summary.scalar('loss', loss_step)
        train_step = tf.train.AdamOptimizer(0.001).minimize(loss_step, global_step=gl_step)

        # 2. Test the gradient tensor operation in the routine of loop,
        # It don't work due to the infinitely nested tensor operation, it is the policy of Tensor Framework
        loss_ta = tf.TensorArray(dtype=tf.float32, size=28)
        loss_ta = loss_ta.unstack(loss)

        init_grad = []
        vars_list = tf.trainable_variables()
        for var in vars_list:
            init_grad.append(tf.zeros_like(var))

        i = tf.constant(0, dtype=tf.int32)

        def condition(i, *args):
            return tf.less(i, 2)

        def loop_fn(i, gradients, all_loss):
            loss_ = all_loss.read(i)
            grads = tf.train.AdamOptimizer(0.001).compute_gradients(loss_, vars_list)
            for idx, (grad, var) in enumerate(grads):
                gradients[idx] += grad
            return i+1, gradients, all_loss

        _, final_grad, _ = tf.while_loop(condition, loop_fn, [i, init_grad, loss_ta])

        train_step = tf.train.AdamOptimizer(0.001).apply_gradients(zip(final_grad, vars_list), global_step=gl_step)

        # return Tensor operation
        return train_step

    def evaluate(out, y):
        correct_prediction = tf.equal(tf.argmax(out, 1), tf.argmax(y, 1))
        acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', acc)
        return acc

    tf.reset_default_graph()
    with tf.Graph().as_default():
        x = tf.placeholder(tf.float32, name='x', shape=[None,784])
        y = tf.placeholder(tf.float32, name='y', shape=[None, 10])

        with tf.variable_scope('mlp_model'):
            output = inference(x, 0.5)

        losses = loss(output, y)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = training(losses, global_step)
        eval_op = evaluate(output, y)
        summary_op = tf.summary.merge_all()
        saver = tf.train.Saver()

        sess = tf.Session()
        summary_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', graph_def=sess.graph_def)
        sess.run(tf.global_variables_initializer())
        valid_errors = []
        pb = ProgressBar()

        for epoch in pb(range(10)):
            avg_loss = 0
            total_batch = int(mnist.train.num_examples/100)

            for i in range(total_batch):
                xs, ys = mnist.train.next_batch(100, False)
                print(xs.shape)
                feed_dict = {x: xs, y: ys}
                sess.run(train_op, feed_dict=feed_dict)
                minibatch_losses = sess.run(losses, feed_dict=feed_dict)
                avg_loss += minibatch_losses / total_batch
            valid_errors.append(avg_loss)

            # display logs per epoch step
            if epoch % 1 == 0:
                val_feed_dict = {x: mnist.validation.images, y: mnist.validation.labels}
                accuracy = sess.run(eval_op, feed_dict=val_feed_dict)
                summary_str = sess.run(summary_op, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, sess.run(global_step))

                save_path = saver.save(sess, 'model-checkpoint')
                print('Model saved in file: %s' % save_path)

    print('Optimization Finished')

    test_feed_dict = {x: mnist.test.images, y: mnist.test.labels}
    accuracy = sess.run(eval_op, feed_dict=test_feed_dict)
    print('Test Accuracy', accuracy)

    plt.plot(np.arange(0, 10, 1), valid_errors, 'ro')
    plt.ylabel('Incurred Error')
    plt.xlabel('Alpha')
    plt.show()


def main(_):
    train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fake_data', nargs='?', const=True, type=bool, default=False, help='Use Fake Data')
    parser.add_argument('--max_steps', type=int, default=1000, help='Number of steps')
    parser.add_argument('--learning rate', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--dropout', type=float, default=0.9, help='Keep prob for training dropout')
    parser.add_argument('--data_dir', type=str, default=os.path.join(os.getcwd(), 'input_data'), help='Storing data')
    parser.add_argument('--log_dir', type=str, default=os.path.join(os.getcwd(), 'logs/mnist_with_summaries'),
                        help='Summaries')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)