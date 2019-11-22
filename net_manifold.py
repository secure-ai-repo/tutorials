
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import argparse
import matplotlib
matplotlib.use('Agg')
from utils.sprial import makeSprial
from utils.plottingutils import *

parser = argparse.ArgumentParser()
parser.add_argument('--numClasses', type=int, required=True, default=2, dest='numClasses', help='number of classes')
parser.add_argument('--numEpochs', type=int, required=True, default=10, dest='numEpochs', help='number of Epochs')
parser.add_argument('--activation', type=str, required=True, default='tanh', dest='activation', help='tanh and ReLU')
args = vars(parser.parse_args())

actDir = {'tanh': tf.tanh, 'ReLU': tf.nn.relu}
numClasses, numEpochs, activation = args['numClasses'], args['numEpochs'], args['activation']
learningRate, L2Reg = 0.0005, 0.005

dataHolder = tf.placeholder(tf.float32, [None, 2]) # data is R2
labelHolder = tf.placeholder(tf.float32, [None, numClasses])

hidden0 = tf.layers.dense(inputs=dataHolder, units=120, activation=activation, name='hidden0')
hidden1 = tf.layers.dense(inputs=hidden0, units=80, activation=activation, name='hidden1')
hidden2 = tf.layers.dense(inputs=hidden1, units=40, activation=activation, name='hidden2')

lastLayer = tf.layers.dense(inputs=hidden2, units=2, activation=activation, name='lastLayer')

outputLogits = tf.layers.dense(inputs=lastLayer, units=numClasses, name='outputLogits')
crossEntropyLoss = tf.losses.softmax_cross_entropy(logits=outputLogits, onehot_labels=labelHolder)
L2RegLoss = L2Reg * tf.add_n([tf.nn.l2_loss(lParameters) for lParameters in tf.trainable_variables()
                              if 'bias' not in lParameters.name])
lossValue = crossEntropyLoss + L2RegLoss

optimizer = tf.train.AdamOptimizer().minimize(lossValue)

correctPrediction = tf.equal(tf.argmax(outputLogits,1), tf.argmax(labelHolder,1))
accuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))

init = tf.global_variables_initializer()

sprialData = list(map(lambda x: makeSprial(x, numClasses), range(numClasses)))
hiddenCheckPoints = set(map(int, np.logspace(np.log10(1), np.log10(numEpochs-1), 350)))

with tf.Session() as sess:
    sess.run(init)

    positionData, labelData = extractElem(0, sprialData), tf.one_hot(extractElem(1,sprialData), depth=numClasses).eval()

    # simple closures to simplify calls to plotting functions and statistics builder
    inputSpace_plotter = wrap_inputSpacePlotter(sess, positionData, dataHolder, labelData, outputLogits, numClasses, args['activation'])
    hiddenLayer_plotter = wrap_inputSpacePlotter(sess, positionData, dataHolder, labelData, lastLayer, numClasses, args['activation'])
    inputToHidden_vectorPlotter = wrap_inputSpacePlotter(sess, positionData, dataHolder, lastLayer, outputLogits, numClasses, args['activation'])
    hiddenStats_builder = wrap_hiddenStatsBuilder(sess, positionData, dataHolder, outputLogits, lastLayer)

    crossEntropyLossValue = []
    for epoch in range(numEpochs):
        _, _lossValue, _crossEntropyLoss, _L2RegLoss = sess.run([optimizer, lossValue, crossEntropyLoss, L2RegLoss],
                                                                {dataHolder: positionData, labelHolder: labelData})
        crossEntropyLoss.append(_crossEntropyLoss)
        _accuracy = accuracy.eval({dataHolder: positionData, labelHolder: labelData})
        hiddenLayer_plotter(epoch, backgroundClassFill=False) if epoch in hiddenCheckPoints else None
        print('%d\t\tcorssEntroy = %f\tL2 = %f\ttotal = %f\taccuracy = %f'
              % (epoch, _crossEntropyLoss, _L2RegLoss, _lossValue, _accuracy))
    print("\nOptimization Finished!\Preparing plots ..\n")

    lossPlotter(numEpochs, crossEntropyLoss, numClasses, args['activation'])
    inputSpace_plotter()
    hiddenLayer_plotter('Final', backgroundClassFill=True)
    inputToHidden_vectorPlotter()

    statDB = hiddenStats_builder() # prepare some statistics about input to last hidden layer function
    anglePlotter(statDB, numClasses, args['activation'])