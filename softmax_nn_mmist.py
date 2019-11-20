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
from tensorflow.python.client import timeline
from tensorflow.examples.tutorials.mnist import input_data


FLAGS = None


def train():
    mnist = input_data.read_data_sets(FLAGS.data_dir, fake_data=FLAGS.fake_data)
    tf.reset_default_graph()
    # graph = tf.Graph()
    sess = tf.InteractiveSession()

    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 784], name='x_input')
        y_ = tf.placeholder(tf.int64, [None], name='y_input')

    with tf.name_scope('input_shape'):
        image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
        tf.summary.image('input', image_shaped_input)

    # def variable_summary(var):
    #     with tf.name_scope('summaries'):
    #         mean = tf.reduce_mean(var)
    #         tf.summary.scalar('mean', mean)
    #         stdev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    #         tf.summary.scalar('stddev', stdev)
    #         tf.summary.scalar('max', tf.reduce_max(var))
    #         tf.summary.scalar('min', tf.reduce_min(var))
    #         tf.summary.histogram('histogram', var)

    def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
        with tf.variable_scope(layer_name):
            #weights_init = tf.truncated_normal([input_dim, output_dim], stddev=0.1)
            # weights = tf.Variable(weights_init)
            weights_stddev = (2.0/tf.cast(input_dim, tf.float32))**0.5
            weights_init = tf.random_normal_initializer(stddev=weights_stddev)
            weights = tf.get_variable('weights', shape=[input_dim, output_dim], initializer=weights_init)
            # variable_summary(weights)
            bias_init = tf.constant(0.0, shape=[output_dim])
            biases = tf.get_variable('biases', initializer=bias_init)
            preactive = tf.matmul(input_tensor, weights) + biases
            activations = act(preactive, name='activation')
            return activations

    def inference(input_layer, keep_pr):
        hidden1 = nn_layer(input_layer, 784, 256, 'layer1')
        hidden2 = nn_layer(hidden1, 256, 256, 'layer2')
        with tf.name_scope('dropped'):
            tf.summary.scalar('dropout_keep_prob', keep_pr)
            dropped = tf.nn.dropout(hidden2, rate=(1 - keep_pr))
        output_layer = nn_layer(dropped, 256, 10, 'output', act=tf.identity)
        return output_layer

    def loss(y_hat, y):
        with tf.name_scope('cross_entropy'):
            cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_hat)
            tf.summary.scalar('cross_entropy', cross_entropy)
        return tf.reduce_mean(cross_entropy)

    keep_prob = tf.placeholder(tf.float32)
    output = inference(x, keep_prob)
    losses = loss(output, y_)

    with tf.name_scope('train'):
        tf.summary.scalar('losses', losses)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        # 1. Test several optimizer operations for understanding their usability
        # train_step = tf.train.AdamOptimizer(0.001).minimize(losses)
        # train_step = tf.train.MomentumOptimizer(learning_rate=0.01,momentum=0.9).minimize(losses,global_step=global_step)
        # train_step = tf.train.AdagradOptimizer(learning_rate=0.01, initial_accumulator_value=0.1,
        #                                       use_locking=False,
        #                                       name='AdaGrad')
        # train_step = tf.train.RMSPropOptimizer(learning_rate=0.01, decay=0.9, momentum=0.0, epsilon=1e-10,
        #                                       use_locking=False,
        #                                       name='RMSProp')
        train_step = tf.train.AdamOptimizer(learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-08,
                                            use_locking=False, name='Adam').minimize(losses, global_step=global_step)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(output, 1), y_)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())

    def feed_dict(train_type):
        if train_type or FLAGS.fake_data:
            xs, ys = mnist.train.next_batch(100, False)
            k = FLAGS.dropout
        else:
            xs, ys = mnist.test.images, mnist.test.labels
            k = 1.0
        return {x: xs, y_: ys, keep_prob: k}

    for i in range(1000):
        if i % 10 == 0:
            xs, ys = mnist.test.images, mnist.test.labels
            k = 1.0
            summary, acc = sess.run([merged, accuracy], feed_dict={x: xs, y_: ys, keep_prob: k})
            test_writer.add_summary(summary, i)
            print('Accuracy at step %s: %s' % (i, acc))
        else:
            if i % 100 == 99:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True),
                                      options=run_options, run_metadata=run_metadata)
                tl = timeline.Timeline(run_metadata.step_stats)
                ctf = tl.generate_chrome_trace_format()
                with open('time_line.jason', 'w') as f:
                    f.write(ctf)
                train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
                train_writer.add_summary(summary, i)
                print('Adding run metadata for', i)
            else:
                xs, ys = mnist.train.next_batch(100, False)
                k = FLAGS.dropout
                summary, _ = sess.run([merged, train_step], feed_dict={x: xs, y_: ys, keep_prob: k})
                train_writer.add_summary(summary, i)

    #saver.save(sess, './logs/mnist_with_summaries', global_step=global_step)
    saver.save(sess, './logs/mnist_with_summaries/model_mnist', global_step=global_step)
    sess.close()
    print('Optimization Finished')

    mnist = input_data.read_data_sets(FLAGS.data_dir, fake_data=FLAGS.fake_data)
    # graph = tf.Graph()
    sess_new = tf.InteractiveSession()

    x = tf.placeholder(tf.float32, [None, 784], name='x_input')
    y_ = tf.placeholder(tf.int64, [None], name='y_input')

    # 2. Test the reuse of the Tensor variables of the saved model
    # ckpt = tf.train.get_checkpoint_state('./logs/model_mnist.ckpt')
    # if ckpt and ckpt.model_checkpoint_path:
    #    print(ckpt.model_checkpoint_path)
    #
    saver = tf.train.Saver()
    if tf.train.checkpoint_exists('./logs/mnist_with_summaries/model_mnist'):
        saver.restore(sess_new, './logs/mnist_with_summaries/model_mnist')
    else:
        sess_new.run(tf.global_variables_initializer())

    with tf.variable_scope('mlp_model') as scope:
        output_opt = inference(x, 0.9)
        cost_opt = loss(output_opt, y_)
        scope.reuse_variables()
        #var_list_opt = []
        # var_list_opt.append(tf.get_variable('layer1/weights')
        var_list_opt = ['layer1/weights', 'layer1/biases',
                        'layer2/weights', 'layer2/biases',
                        'output/weights', 'output/biases']
        var_list_opt = [tf.get_variable(v) for v in var_list_opt]

    with tf.variable_scope('mlp_init') as scope:
        output_rand = inference(x, 0.9)
        cost_rand = loss(output_rand, y_)
        scope.reuse_variables()
        var_list_rand = ['layer1/weights', 'layer1/biases',
                         'layer2/weights', 'layer2/biases',
                         'output/weights', 'output/biases']
        var_list_rand = [tf.get_variable(v) for v in var_list_rand]
        #sess.run(tf.initialize_variables(var_list_rand))

    global_var = tf.global_variables()
    is_var_init = [tf.is_variable_initialized(v) for v in global_var]
    is_initialized = sess_new.run(is_var_init)
    not_initialized_var = [var for (var, init) in zip(global_var, is_initialized) if not init]
    if len(not_initialized_var):
        sess_new.run(tf.variables_initializer(not_initialized_var))

    with tf.variable_scope('mlp_inter') as scope:
        alpha = tf.placeholder('float', [1, 1])
        beta = 1 - alpha
        h1_W_inter = var_list_opt[0] * beta + var_list_rand[0] * alpha
        h1_b_inter = var_list_opt[1] * beta + var_list_rand[1] * alpha
        h2_W_inter = var_list_opt[2] * beta + var_list_rand[2] * alpha
        h2_b_inter = var_list_opt[3] * beta + var_list_rand[3] * alpha
        o_W_inter = var_list_opt[4] * beta + var_list_rand[4] * alpha
        o_b_inter = var_list_opt[5] * beta + var_list_rand[5] * alpha
        h1_inter = tf.nn.relu(tf.matmul(x, h1_W_inter) + h1_b_inter)
        h2_inter = tf.nn.relu(tf.matmul(h1_inter, h2_W_inter) + h2_b_inter)
        o_inter = tf.nn.relu(tf.matmul(h2_inter, o_W_inter) + o_b_inter)
        losses_inter = loss(o_inter, y_)

    summary_writer = tf.summary.FileWriter('./linear_inter_logs', graph_def=sess.graph_def)
    summary_op = tf.summary.merge_all()

    results = []
    for a in np.arange(-2, 2, 0.01):
        feed_dict = {x: mnist.test.images, y_: mnist.test.labels, alpha: [[a]]}
        losses = sess_new.run([losses_inter], feed_dict=feed_dict)
        #losses, summary_str = sess_new.run([losses_inter], feed_dict=feed_dict)
        #summary_str = sess.run([summary_op], feed_dict=feed_dict)
        #summary_writer.add_summary(summary_str, (a + 2) / 0.01)
        results.append(losses)

    train_writer.close()
    train_writer.close()

    plt.plot(np.arange(-2, 2, 0.01), results, 'ro')
    plt.ylabel('Incurred Error')
    plt.xlabel('Alpha')
    plt.show()


    # 3. Test the random walk of steps in gradient decents
    # step_range = 10
    # momentum = 0.9
    # step_choices = range(-1 * step_range, step_range + 1)
    # rand_walk = [np.random.choice(step_choices) for x in range(100)]
    # momentum_rand_walk = [np.random.choice(step_choices)]
    # for i in range(len(rand_walk) - 1):
    #     prev = momentum_rand_walk[-1]
    #     rand_choice = np.random.choice(step_choices)
    #     new_step = momentum * prev + (1-momentum) * rand_choice
    #     momentum_rand_walk.append(new_step)


def main(_):
    train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fake_data', nargs='?', const=True, type=bool, default=False, help='Use Fake Data')
    parser.add_argument('--max_steps', type=int, default=1000, help='Number of steps')
    parser.add_argument('--learning rate', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--dropout', type=float, default=0.9, help='Keep prob for training dropout')
    parser.add_argument('--data_dir', type=str, default=os.path.join(os.getcwd(), 'input_data'), help='Storing data')
    parser.add_argument('--log_dir', type=str, default=os.path.join(os.getcwd(), 'logs/mnist_with_summaries'), help='Summaries')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]]+unparsed)
