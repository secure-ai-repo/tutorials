"""
For the purpose of upgrading the development of Tensorflow AI Vision Project
Edited by Dr. Hyeung-yun Kim


Train the model on dataset
"""
import sys
import json
import os
from os.path import basename
import tensorflow as tf
import numpy as np
import cv2
import pickle
import time
import math
from multiprocessing.pool import ThreadPool
from time import time as timer
from YOLO.defaults import argHandler
from YOLO.net.tensorop import op_types, identity
from YOLO.net.tensorop import HEADER, LINE
from YOLO.net import yolo
from YOLO.net import yolov2
from YOLO.net.basecnn import BaseCNN, create_layerop

from tensorflow.python import debug as tf_debug


FLAGS = argHandler()
FLAGS.setDefaults()
FLAGS.parseArgs(sys.argv)

# make sure all necessary dirs exist
requiredDirectories = [FLAGS.imgdir, FLAGS.binary, FLAGS.backup, os.path.join(FLAGS.imgdir, 'out')]
if FLAGS.summary:
    requiredDirectories.append(FLAGS.summary)

for d in requiredDirectories:
    this = os.path.abspath(os.path.join(os.path.curdir, d))
    if not os.path.exists(this): os.makedirs(this)

## ========================================= ##
class loader(object):
    """
    interface to work with both .weights and .ckpt files in loading / recollecting / resolving mode
    """
    VAR_LAYER = ['convolutional', 'connected', 'local', 'select', 'conv-select', 'extract', 'conv-extract']

    def __init__(self, *args):
        self.src_key = list()
        self.vals = list()
        self.load(*args)

    def __call__(self, key):
        for idx in range(len(key)):
            val = self.find(key, idx)
            if val is not None: return val
        return None

    def find(self, key, idx):
        up_to = min(len(self.src_key), 4)
        for i in range(up_to):
            key_b = self.src_key[i]
            if key_b[idx:] == key[idx:]:
                return self.yields(i)
        return None

    def yields(self, idx):
        del self.src_key[idx]
        temp = self.vals[idx]
        del self.vals[idx]
        return temp

    def load(self, param): pass


class weights_loader(loader):
    """one who understands .weights files"""
    _W_ORDER = dict({  # order of param flattened into .weights file
        'convolutional': ['biases', 'gamma', 'moving_mean', 'moving_variance', 'kernel'],
        'connected': ['biases', 'weights'],
        'local': ['biases', 'kernels']
    })

    def load(self, path, src_layers):
        self.src_layers = src_layers
        walker = weights_walker(path)

        for i, layer in enumerate(src_layers):
            if layer.type not in self.VAR_LAYER: continue
            self.src_key.append([layer])

            if walker.eof:
                new = None
            else:
                args = layer.signature
                new = create_layerop(*args)
            self.vals.append(new)

            if new is None: continue
            order = self._W_ORDER[new.type]
            for par in order:
                if par not in new.wshape: continue
                val = walker.walk(new.wsize[par])
                new.w[par] = val
            new.finalize(walker.transpose)

        if walker.path is not None:
            assert walker.offset == walker.size, 'expect {} bytes, found {}'.format(walker.offset, walker.size)
            print('Successfully identified {} bytes'.format(walker.offset))


class weights_walker(object):
    """incremental reader of float32 binary files"""
    def __init__(self, path):
        self.eof = False  # end of file
        self.path = path  # current pos
        if path is None:
            self.eof = True
            return
        else:
            self.size = os.path.getsize(path)  # save the path
            major, minor, revision, seen = np.memmap(path, shape=(), mode='r', offset=0, dtype='({})i4,'.format(4))
            self.transpose = major > 1000 or minor > 1000
            self.offset = 16

    def walk(self, size):
        if self.eof: return None
        end_point = self.offset + 4 * size
        assert end_point <= self.size, 'Over-read {}'.format(self.path)
        float32_1D_array = np.memmap(self.path, shape=(), mode='r', offset=self.offset, dtype='({})float32,'.format(size))
        self.offset = end_point
        if end_point == self.size:
            self.eof = True
        return float32_1D_array

## =========================================== ##
class YOLOv2(object):
    def __init__(self, meta, FLAGS):
        # self.constructor(meta, FLAGS)
        def _to_color(indx, base):
            """ return (b, r, g) tuple"""
            base2 = base * base
            b = 2 - indx / base2
            r = 2 - (indx % base2) / base
            g = 2 - (indx % base2) % base
            return (b * 127, r * 127, g * 127)

        if 'labels' not in meta:
            yolo.labels(meta, FLAGS)  # We're not loading from a .pb so we do need to load the labels
        assert len(meta['labels']) == meta['classes'], (
                'labels.txt and {} indicate' + ' inconsistent class numbers').format(meta['model'])

        # assign a color for each label
        colors = list()
        base = int(np.ceil(pow(meta['classes'], 1. / 3)))
        for x in range(len(meta['labels'])):
            colors += [_to_color(x, base)]
        meta['colors'] = colors
        self.fetch = list()
        self.meta, self.FLAGS = meta, FLAGS

        # over-ride the threshold in meta if FLAGS has it.
        if FLAGS.threshold > 0.0:
            self.meta['thresh'] = FLAGS.threshold

    # image data constructor
    parse = yolo.parse                  # annotation xml
    shuffle = yolo.shuffle              # numpy batch data
    preprocess = yolo.preprocess        # open cv image processing
    loss = yolov2.loss                  # tensor data
    is_inp = yolo.is_inp                # image type
    _batch = yolov2._batch              # open cv image processing
    resize_input = yolo.resize_input
    findboxes = yolov2.findboxes
    process_box = yolo.process_box

## ========= MY_YOLO Class's Methods ============ ##


def train(self):

    loss_ph = self.framework.placeholders       # MY_YOLO.YOLO2.placeholders
    loss_mva = None; profile = list()

    batches = self.framework.shuffle()
    loss_op = self.framework.loss               # MY_YOLO.YOLO2.loss (not loss() function)

    for i, (x_batch, datum) in enumerate(batches):
        if not i:
            print('Training statistics: Learning rate : {}, Batch size : {}, Epoch number : {}, Backup every : {}'
                  .format(self.FLAGS.lr, self.FLAGS.batch, self.FLAGS.epoch, self.FLAGS.save))

        """ loss_feed_val = {'probs': probs, 'confs': confs,'coord': coord, 'proid': proid,'areas': areas, 'upleft': upleft,'botright': botright} """
        """ self.placeers = {'probs': _probs, 'confs': _confs, 'coord': _coord, 'proid': _proid,'areas': _areas, 'upleft': _upleft, 'botright': _botright} """

        # import pudb; pudb.set_trace()
        feed_dict = {loss_ph[key]: datum[key] for key in loss_ph}   # feed_dict = {palceholder1 : np_array1}
        feed_dict[self.inp] = x_batch     # self.inp = tf.placeholder(tf.float32, inp_size, 'input') : inp_feed_val=img
        feed_dict.update(self.feed)       # add self.feed dictionary to feed_dict dictionary

        fetches = [self.train_op, loss_op]

        if self.FLAGS.summary:
            fetches.append(self.summary_op)     # tf.summary.merge_all()
        # ============================= #
        fetched = self.sess.run(fetches, feed_dict)     # self.sess.partial_run()
        loss = fetched[1]                       # loss = .5 * tf.reduce_mean(loss)

        if loss_mva is None: loss_mva = loss
        loss_mva = .9 * loss_mva + .1 * loss    # moving average
        step_now = self.FLAGS.load + i + 1

        if self.FLAGS.summary:
            self.writer.add_summary(fetched[2], step_now) # write summary

        print('step {} - loss {} - moving ave loss {}'.format(step_now, loss, loss_mva))
        profile += [(loss, loss_mva)]


pool = ThreadPool()


def savepb(self):
    """
    Create a standalone const graph def So that C++ can load and run it.
    """
    basenet_ckpt = self.basenet
    flags_ckpt = self.FLAGS
    flags_ckpt.savepb = None  # signal

    with self.graph.as_default() as g:
        for var in tf.trainable_variables():
            print(var.name)
            name = ':'.join(var.name.split(':')[:-1])
            var_name = name.split('-')
            val = var.eval(self.sess)
            l_idx = int(var_name[0])
            w_sig = var_name[-1]
            trained = var.eval(self.sess)
            basenet_ckpt.layers[l_idx].w[w_sig] = trained

    for layer in basenet_ckpt.layers:
        for ph in layer.h:  # Set all placeholders to dfault val
            layer.h[ph] = self.feed[layer.h[ph]]['dfault']

    myyolov2_ckpt = MY_YOLO2(basenet=basenet_ckpt, FLAGS=self.FLAGS)
    myyolov2_ckpt.sess = tf.Session(graph=myyolov2_ckpt.graph)
    # myyolov2_ckpt.predict() # uncomment for unit testing

    name = 'graph-{}.pb'.format(self.meta['model'])
    print('Saving const graph def to {}'.format(name))
    graph_def = myyolov2_ckpt.sess.graph_def
    tf.train.write_graph(graph_def, './', name, False)


def predict(self):
    inp_path = self.FLAGS.imgdir
    all_inps = os.listdir(inp_path)
    all_inps = [i for i in all_inps if self.framework.is_inp(i)]

    if not all_inps:
        msg = 'Failed to find any images in {} .'
        exit('Error: {}'.format(msg.format(inp_path)))

    batch = min(self.FLAGS.batch, len(all_inps))

    # predict in batches
    n_batch = int(math.ceil(len(all_inps) / batch))
    for j in range(n_batch):
        from_idx = j * batch
        to_idx = min(from_idx + batch, len(all_inps))

        # collect images input in the batch
        this_batch = all_inps[from_idx:to_idx]
        inp_feed = pool.map(lambda inp: (np.expand_dims(self.framework.preprocess(os.path.join(inp_path, inp)), 0)), this_batch)

        # Feed to the net
        feed_dict = {self.inp: np.concatenate(inp_feed, 0)}
        print('Forwarding {} inputs ...'.format(len(inp_feed)))
        start = time.time()
        out = self.sess.run(self.out, feed_dict)
        stop = time.time(); last = stop - start
        print('Total time = {}s / {} inps = {} ips'.format(last, len(inp_feed), len(inp_feed) / last))


def say(self, *msgs):
    if not self.FLAGS.verbalise:
        return
    msgs = list(msgs)
    for msg in msgs:
        if msg is None: continue
        print(msg)


## =========================================== ##

class MY_YOLO2(object):
    _TRAINER = dict({
        'rmsprop': tf.train.RMSPropOptimizer,
        'adadelta': tf.train.AdadeltaOptimizer,
        'adagrad': tf.train.AdagradOptimizer,
        'adagradDA': tf.train.AdagradDAOptimizer,
        'momentum': tf.train.MomentumOptimizer,
        'adam': tf.train.AdamOptimizer,
        'sgd': tf.train.GradientDescentOptimizer
    })
    # Class methods
    say = say
    train = train
    savepb =savepb
    predict = predict

    def __init__(self, FLAGS, basenet=None):
        self.ntrain = 0
        self.FLAGS = FLAGS
        if self.FLAGS.pbLoad and self.FLAGS.metaLoad:
            print('\nLoading from .pb and .meta')
            self.graph = tf.Graph()

        if basenet is None:
            basenet = BaseCNN(FLAGS)   # <<==== initialize CNN (DarkNet) parameters
            self.ntrain = len(basenet.layers)

        self.num_layer = len(basenet.layers)
        model = basename(basenet.meta['model'])
        basenet.meta['name'] = '.'.join(model.split('.')[:-1])
        self.meta = basenet.meta
        self.basenet = basenet

        self.framework = YOLOv2(basenet.meta, FLAGS)  # <<=== initialize YOLOv2 parameters

        print('\nBuilding net ...')
        start = time.time()
        self.graph = tf.Graph()

        with self.graph.as_default() as g:
            # Placeholders
            inp_size = [None] + basenet.meta['inp_size']
            self.inp = tf.placeholder(tf.float32, inp_size, 'input')  # tensor-input fetching
            self.feed = dict()  # other placeholders
            # import ipdb; ipdb.set_trace()  # XXX_BREAKPOINT

            # Build the forward pass
            state = identity(self.inp)
            roof = self.num_layer - self.ntrain
            self.say(HEADER, LINE)
            for i, layer in enumerate(basenet.layers):
                args = [layer, state, i, roof, self.feed]  # chaining tensor operations
                layer_type = list(args)[0].type
                state = op_types[layer_type](*args)  # tensor-class instance
                mess = state.verbalise()
                self.say(mess)
            self.say(LINE)

            self.top = state
            self.out = tf.identity(state.out, name='output')   # tensor-output instance fetching

            if self.FLAGS.train:
                self.framework.loss(self.out)  # pass the CNN output to YOLOv2
                print('Building {} train op'.format(basenet.meta['model']))

                optimizer = self._TRAINER[self.FLAGS.trainer](self.FLAGS.lr)
                gradients = optimizer.compute_gradients(self.framework.loss)
                self.train_op = optimizer.apply_gradients(gradients)

            if self.FLAGS.summary:
                self.summary_op = tf.summary.merge_all()
                self.writer = tf.summary.FileWriter(self.FLAGS.summary + 'train')

            cfg = dict({'allow_soft_placement': False, 'log_device_placement': False})
            self.sess = tf.Session(config=tf.ConfigProto(**cfg))
            # self.sess = tf_debug.LocalCLIDebugWrapperSession(self.sess)

            self.sess.run(tf.global_variables_initializer())
            if not self.ntrain:
                return

            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=self.FLAGS.keep)
            if self.FLAGS.summary:
                self.writer.add_graph(self.sess.graph)

        print('Finished in {}s\n'.format(time.time() - start))

## ============================================= ##
tfnet = MY_YOLO2(FLAGS)

if FLAGS.train:
    print('Enter training ...');
    tfnet.train()

if FLAGS.savepb:
    print('Rebuild a constant version ...')
    tfnet.savepb(); exit('Done')

tfnet.predict()


