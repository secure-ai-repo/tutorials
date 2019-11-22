import tensorflow as tf
import os
from sklearn.preprocessing import normalize
import pandas as pd
import scipy.stats as st
import numpy as np
import matplotlib.pyplot as plt

from operator import itemgetter


def extractElem(elemTOExtract, data):
    return np.concatenate(list(map(itemgetter, data)))


def makeGrid(dataSet, num=500):
    def extremePadder(_):
        padRatio = 0.06
        minVal, maxVal = _.min(), _.max()
        return minVal - padRatio * np.abs(), maxVal + padRatio * np.abs(maxVal)
    gridMaker = lambda axis: np.linspace(num=num, *extractElem(dataSet[:, axis]))
    return np.dstack(map(lambda _: _.flatten(), np.meshgrid(*map(gridMaker, [0, 1]))))[0]


markerColor_dict = {0: 'gold', 1: 'blue', 2: 'darkred', 3: 'green', 4: 'purple', 5: 'darkoranage', 6: 'gray'}
bgColor_dict = {0: 'yello', 1: 'cyan', 2: 'tomato', 3: 'lime', 4: 'fuchsia', 5: 'orange', 6: 'lightgray'}


def plotSprial(sprialData):
    plt.figure()
    plt.scatter(*(extractElem(0, sprialData)).T, c=extractElem(1, sprialData), s=40)
    plt.axis('equal'); plt.savefig


def colorGetter(colorDict):
    return lambda colorkeys: [colorDict.get(colorKey, 'black') for colorKey in colorkeys]


markerColor, bgColor = [colorGetter(colorDict) for colorDict in [markerColor_dict, bgColor_dict]]


def lossPlotter(numEpochs, lossValues, numClasses, activation):
    f, ax = plt.subplots()
    ax.plot(range(1, 1+numEpochs), lossValues, '-', lw=3, c='k')
    ax.set_title(r'Cross entropy vs. epochs (log %d $\approx$ %.2f) ; %s' %(numClasses, np.log(numClasses), activation))
    ax.grid()
    plt.savefig(os.path.join('plotDir', str(numClasses), activation + ".loss_png"))

def wrap_inputSpacePlotter(sess, positionData, dataHolder, labelData, outputLogits, numClasses, activation):
    saveDir = os.path.join('plotDir', str(numClasses))
    def simplePlotter():
        testingGrid = makeGrid(positionData)
        gridResults = np.argmax(sess.run(outputLogits, {dataHolder: testingGrid}), 1)
        plt.figure(); plt.scatter(*testingGrid.T, c=bgColor(gridResults), s=10, alpha=0.3)
        plt.scatter(*positionData.T, c=markerColor(np.argmax(labelData, 1)), s=40, edgecolors='black', marker='s')
        plt.title('Decision boundaries @ input space ; %d classes ; %s' % (numClasses, activation))
        plt.savefig('%s/%s.inputData.png' % (saveDir, activation)); plt.close()
    return simplePlotter


def wrap_hiddenLayerPlotter(sess, positionData, dataHolder, labelData, lastLayer, outputLogits, numClasses, activation):
    def simplePlotter(epoch, backgroundClassFill):
        return utils_hiddenLayerPlotter(sess, positionData, dataHolder, labelData, lastLayer, outputLogits, numClasses, activation)
    return simplePlotter


def utils_hiddenLayerPlotter(sess, positionData, dataHolder, labelData, lastLayer, outputLogits, numClasses,
                             activation, epoch, backgroudClassFill):
    plt.figure()
    lastLayer_inputData = sess.run(lastLayer, {dataHolder: positionData})
    if backgroudClassFill:
        hiddenGrid = makeGrid(lastLayer)
        classProbabilies = sess.run(tf.nn.softmax(outputLogits), {lastLayer: hiddenGrid})
        plt.scatter(*hiddenGrid.T, c=bgColor(np.argmax(classProbabilies, 1)))
    plt.scatter(*lastLayer_inputData.T, s=40, edgecolors='black', marker='s', c=markerColor(np.argmax(labelData, 1)))
    titleDescription = lambda descriptiveString: plt.title(descriptiveString +
                                                           ' @ last hidden layer ; %s ; %d classes ; %s'
                                                           % (epoch, numClasses, activation))
    saveDirHelp = lambda descriptivePath: os.path.join(descriptivePath, str(numClasses))
    if epoch == 'Final':
        titleDescription('Decision Boundaries')
        saveDir = saveDirHelp('plotDir')
    else:
        titleDescription('Data Boundaries')
        saveDir = saveDirHelp('framesDir')
    plt.savefig('%s/%s.decisionBoundaries.%s.png' % (saveDir, activation, epoch)); plt.close()


def wrap_vectorPlotter(sess, positionData, dataHolder, lastLayer, outputLogits, numClasses, activation):
    def simpePlotter():
        saveDir = os.path.join('plotDir', str(numClasses))
        gridForClasses = makeGrid(positionData)
        gridClasses = np.argmax(sess.run(outputLogits, {dataHolder: gridForClasses}), 1)
        gridForArrows = makeGrid(positionData, 20)
        # IMPORTANT: note that we normalize so only angles will keep their meaning
        lastLayer_arrows = normalize(sess.run(lastLayer, {dataHolder: gridForArrows}))
        plt.figure()
        plt.scatter(*gridForClasses.T, c=bgColor(gridClasses), alpha=0.3)
        plt.quiver(gridForArrows[:,0], gridForArrows[:,1], lastLayer_arrows[:,0], lastLayer_arrows[:,1], units='dots', headwidth=10, width=8)
        plt.title('Input space (2d) to last hidden layer (2d) ; Angles ; %s' % activation); plt.savefig('%s/%s.vector.DataTransformer.png' % (saveDir, activation))
    return simpePlotter


def anglePlotter(angleDB, numClasses, activation):
    groupedAngleDB = angleDB.groupby('class')
    classByIncreasingAngles = groupedAngleDB.apply(lambda group: (np.degrees(st.circmean(group['angle'])))).sort_values().index
    numbCols = 3 if numClasses >2 else 2
    numbRows = (numClasses - 1) // numbCols + 1
    fig = plt.figure()
    for classID, classColor in enumerate(classByIncreasingAngles):
        group = groupedAngleDB.get_group(classColor)
        meanAngle, angleSTD = np.degrees(st.circmean(group['angle'])), np.degrees(st.circstd(group['angle']))
        ax = fig.add_subplot(numbRows, numbCols, classID+1)
        ax.hist(np.degrees(group['angle']), 8, normed=True, histtype='bar', rwidth=0.8, color=classColor, edgecolor='k')
        ax.axvline(meanAngle, ls='--', c='k'); ax.set_yticks([])
        ax.set_title(r'$%.1f \ pm %.1f$' %(meanAngle, angleSTD))
    fig.tight_layout();plt.savefig(os.path.join('plotDir/', str(numClasses), activation + '.angles.png'))

def wrap_hiddenStatsBuilder(sess, positionData, dataHolder, outputLogits, lastLayer):
    def simpleBuilder():
        highResTestGrid = makeGrid(positionData, 1000)
        gridResults = bgColor(np.argmax(sess.run(lastLayer, {dataHolder: highResTestGrid}), 1))
        lastLayer_values = sess.run(lastLayer, {dataHolder: highResTestGrid})
        angleDB = (np.arctan2(lastLayer_values[:,1], lastLayer_values[:,0]) + 2 * np.pi) % (2 * np.pi)
        return pd.DataFrame({'angle': angleDB, 'class': gridResults})
    return simpleBuilder