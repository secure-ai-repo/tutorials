
import tensorflow as tf
from cifar import Cifar
import model
import helper
from tqdm import tqdm


def train():

    learing_rate = 0.001
    batch_size = 16
    num_epoches = 10
    dropout_rate = 0.8

    y = tf.placeholder(tf.float32, [None, model.n_classes])
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model.out, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learing_rate).minimize(cost)
    correct_pred = tf.equal(tf.argmax(model.out, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    tf.summary.histogram('cost', cost)
    tf.summary.histogram('accuracy', accuracy)

    cifar = Cifar(batch_size=batch_size)

    init = tf.initialize_all_variables()

    saver = tf.train.Saver()
    i = 0

    with tf.Session() as sess:
        sess.run(init)

        # Create reader and writer, which writes to ./logs folder
        reader = tf.WholeFileReader()
        writer = tf.summary.FileWriter('./logs/', sess.graph)

        for epoch in range(num_epoches):
            for batch in tqdm(cifar.batches, desc="Epoch {}".format(epoch), unit="batch"):
                #for batch in cifar.batches:
                inp, out = helper.transform_to_input_output(batch, dim=model.n_classes)
                sess.run([optimizer], feed_dict={model.input_images: inp, y: out, model.dropout: dropout_rate})

            merge = tf.summary.merge_all()
            acc, loss, summary = sess.run([accuracy, cost, merge],
                                          feed_dict={model.input_images: inp, y: out, model.dropout: 1.})
            writer.add_summary(summary, i)
            i = i + 1
            print("Acc: {} Loss {}".format(acc, loss))

            inp_test, out_test = helper.transform_to_input_output(cifar.test_set, dim=model.n_classes)
            test_acc = sess.run([accuracy],
                                feed_dict={model.input_images: inp_test, y: out_test, model.dropout: 1.})
            print("Test Acc: {}".format(test_acc))

            saver.save(sess, './saved_model/alexnet.ckpt')


def main(_):
    train()


if __name__ == '__main__':
    tf.app.run(main=main)
