import tensorflow as tf
import os
import numpy as np
import re
from tensorflow.python.platform import gfile


def save_graph(inference_func, checkpoint_path, image_size, dest_path):
    """ Saves a computation graph with all it's variables.
    :param inference_func: the function that builds the graph and gets image input
    :param checkpoint_path: the path to a checkpoint containing values of variables
    :param image_size: the image input size
    :param dest_path: file path to save the graph definition protocol buffer
    :return:
    """

    g = tf.Graph()
    vars = {}
    with g.as_default():
        with tf.Session() as sess:
            d = np.ones([1, conf.parts_avg_size[5][1], conf.parts_avg_size[5][0],3], dtype=np.float32)

            input_data = tf.placeholder(tf.float32,shape=[1,image_size[1],image_size[0],3], name="input_placeholder")
            logits = inference_func(input_data)

            init = tf.initialize_all_variables()
            sess.run(init)
            saver = tf.train.Saver(tf.trainable_variables(),max_to_keep=0)
            saver.restore(sess, checkpoint_path)

            print(sess.run(logits, {input_data: d}))
            for v in tf.trainable_variables():
                # vars[v.name] = sess.run(v) # <== causing error
                vars[v.value().name] = sess.run(v) # <== solution

    g2 = tf.Graph()
    consts = {}
    with g2.as_default():
        with tf.Session() as sess:
            for k in vars.keys():
                consts[k] = tf.constant(vars[k])

            tf.import_graph_def(g2.as_graph_def(), input_map={name: consts[name] for name in consts.keys()}) # <== error

            tf.train.write_graph(sess.graph_def, dest_path, 'graph.pbtxt', False)

    return os.path.join(dest_path,'graph.pbtxt')

# ============================================
# (1) Train

# (2) Create nodes to freeze trained weights/biases
def _freeze():
    regex = re.compile('^[^:]*')
    with tf.name_scope('assign_ops'):
       for tvar in tf.trainable_variables():
           tf.assign(tvar, tvar.eval(), name=re.match(regex, tvar.name).group(0))

# (3) Save graph
with open(outfile, 'wb') as f:
    f.write(sess.graph_def.SerializeToString())

# (4) (Later) restore graph and get handles for desired nodes
with open(graph_def, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())


x, dropout, predictions = tf.import_graph_def(graph_def, return_elements=['x:0', 'dropout:0', 'predictions:0'], name='')


# (5) Re-assign stored values
with tf.Session(config=config) as sess:
    assign_ops = [op for op in tf.Graph.get_operations(sess.graph) if 'assign_ops' in op.name]
    sess.run(assign_ops)

