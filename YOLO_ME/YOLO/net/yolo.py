"""
For the purpose of upgrading the development of Tensorflow AI Vision Project
Edited by Dr. Hyeung-yun Kim


YOLO architecture on Base Neural Network
"""
import tensorflow.contrib.slim as slim
import tensorflow as tf
import cv2
import os
import json
from YOLO.utils.box import BoundBox, box_iou, prob_compare
from YOLO.utils.im_transform import imcv2_recolor, imcv2_affine_trans
from YOLO.utils.yolo2_findboxes import box_constructor
from YOLO.utils.pascal_voc_clean_xml import pascal_voc_clean_xml
from numpy.random import permutation as perm
from copy import deepcopy
import numpy as np

""" YOLO framework __init__ equivalent"""


def constructor(self, meta, FLAGS):
    def _to_color(indx, base):
        """ return (b, r, g) tuple"""
        base2 = base * base
        b = 2 - indx / base2
        r = 2 - (indx % base2) / base
        g = 2 - (indx % base2) % base
        return (b * 127, r * 127, g * 127)

    if 'labels' not in meta:
        labels(meta, FLAGS)  # We're not loading from a .pb so we do need to load the labels
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


## ============================================= #
def loss(self, net_out):
    """ Takes net.out and placeholders value returned in batch() func above, to build train_op and loss """
    # meta
    m = self.meta
    sprob = float(m['class_scale'])
    sconf = float(m['object_scale'])
    snoob = float(m['noobject_scale'])
    scoor = float(m['coord_scale'])
    S, B, C = m['side'], m['num'], m['classes']
    SS = S * S  # number of grid cells

    print('{} loss hyper-parameters:'.format(m['model']))
    print('\tside    = {}'.format(m['side']))
    print('\tbox     = {}'.format(m['num']))
    print('\tclasses = {}'.format(m['classes']))
    print('\tscales  = {}'.format([sprob, sconf, snoob, scoor]))

    size1 = [None, SS, C]
    size2 = [None, SS, B]

    # return the below placeholders
    _probs = tf.placeholder(tf.float32, size1)
    _confs = tf.placeholder(tf.float32, size2)
    _coord = tf.placeholder(tf.float32, size2 + [4])
    # weights term for L2 loss
    _proid = tf.placeholder(tf.float32, size1)
    # material calculating IOU
    _areas = tf.placeholder(tf.float32, size2)
    _upleft = tf.placeholder(tf.float32, size2 + [2])
    _botright = tf.placeholder(tf.float32, size2 + [2])

    self.placeholders = {
        'probs': _probs, 'confs': _confs, 'coord': _coord, 'proid': _proid,
        'areas': _areas, 'upleft': _upleft, 'botright': _botright
    }

    # Extract the coordinate prediction from net.out
    coords = net_out[:, SS * (C + B):]
    coords = tf.reshape(coords, [-1, SS, B, 4])
    wh = tf.pow(coords[:, :, :, 2:4], 2) * S  # unit: grid cell
    area_pred = wh[:, :, :, 0] * wh[:, :, :, 1]  # unit: grid cell^2
    centers = coords[:, :, :, 0:2]  # [batch, SS, B, 2]
    floor = centers - (wh * .5)  # [batch, SS, B, 2]
    ceil = centers + (wh * .5)  # [batch, SS, B, 2]

    # calculate the intersection areas
    intersect_upleft = tf.maximum(floor, _upleft)
    intersect_botright = tf.minimum(ceil, _botright)
    intersect_wh = intersect_botright - intersect_upleft
    intersect_wh = tf.maximum(intersect_wh, 0.0)
    intersect = tf.multiply(intersect_wh[:, :, :, 0], intersect_wh[:, :, :, 1])

    # calculate the best IOU, set 0.0 confidence for worse boxes
    iou = tf.truediv(intersect, _areas + area_pred - intersect)
    best_box = tf.equal(iou, tf.reduce_max(iou, [2], True))
    best_box = tf.to_float(best_box)
    confs = tf.multiply(best_box, _confs)

    # take care of the weight terms
    conid = snoob * (1. - confs) + sconf * confs
    weight_coo = tf.concat(4 * [tf.expand_dims(confs, -1)], 3)
    cooid = scoor * weight_coo
    proid = sprob * _proid

    # flatten 'em all
    probs = slim.flatten(_probs)
    proid = slim.flatten(proid)
    confs = slim.flatten(confs)
    conid = slim.flatten(conid)
    coord = slim.flatten(_coord)
    cooid = slim.flatten(cooid)

    self.fetch += [probs, confs, conid, cooid, proid]
    true = tf.concat([probs, confs, coord], 1)
    wght = tf.concat([proid, conid, cooid], 1)
    print('Building {} loss'.format(m['model']))
    loss = tf.pow(net_out - true, 2)
    loss = tf.multiply(loss, wght)
    loss = tf.reduce_sum(loss, 1)
    self.loss = .5 * tf.reduce_mean(loss)
    tf.summary.scalar('{} loss'.format(m['model']), self.loss)


## ============================================= #
def _fix(obj, dims, scale, offs):
    for i in range(1, 5):
        dim = dims[(i + 1) % 2]
        off = offs[(i + 1) % 2]
        obj[i] = int(obj[i] * scale - off)
        obj[i] = max(min(obj[i], dim), 0)


def resize_input(self, im):
    h, w, c = self.meta['inp_size']
    imsz = cv2.resize(im, (w, h))
    imsz = imsz / 255.
    imsz = imsz[:, :, ::-1]
    return imsz


def process_box(self, b, h, w, threshold):
    max_indx = np.argmax(b.probs)
    max_prob = b.probs[max_indx]
    label = self.meta['labels'][max_indx]
    if max_prob > threshold:
        left = int((b.x - b.w / 2.) * w)
        right = int((b.x + b.w / 2.) * w)
        top = int((b.y - b.h / 2.) * h)
        bot = int((b.y + b.h / 2.) * h)
        if left < 0:  left = 0
        if right > w - 1: right = w - 1
        if top < 0: top = 0
        if bot > h - 1:  bot = h - 1
        mess = '{}'.format(label)
        return (left, right, top, bot, mess, max_indx, max_prob)
    return None


def findboxes(self, net_out):
    meta, FLAGS = self.meta, self.FLAGS
    threshold = FLAGS.threshold

    boxes = []
    # boxes = yolo_box_constructor(meta, net_out, threshold)
    boxes = box_constructor(meta, net_out, threshold)

    return boxes


def preprocess(self, im, allobj=None):
    """
    Takes an image, return it as a numpy tensor that is readily
    to be fed into tfnet. If there is an accompanied annotation (allobj),
    meaning this preprocessing is serving the train process, then this
    image will be transformed with random noise to augment training data,
    using scale, translation, flipping and recolor. The accompanied
    parsed annotation (allobj) will also be modified accordingly.
    """
    if type(im) is not np.ndarray:
        im = cv2.imread(im)

    if allobj is not None:  # in training mode
        result = imcv2_affine_trans(im)
        im, dims, trans_param = result
        scale, offs, flip = trans_param
        for obj in allobj:
            _fix(obj, dims, scale, offs)
            if not flip: continue
            obj_1_ = obj[1]
            obj[1] = dims[0] - obj[3]
            obj[3] = dims[0] - obj_1_
        im = imcv2_recolor(im)

    im = self.resize_input(im)
    if allobj is None: return im
    return im  # , np.array(im) # for unit testing


def postprocess(self, net_out, im, save=True):
    """
    Takes net output, draw predictions, save to disk
    """
    meta, FLAGS = self.meta, self.FLAGS
    threshold = FLAGS.threshold
    colors, labels = meta['colors'], meta['labels']

    boxes = self.findboxes(net_out)

    if type(im) is not np.ndarray:
        imgcv = cv2.imread(im)
    else:
        imgcv = im

    h, w, _ = imgcv.shape
    resultsForJSON = []
    for b in boxes:
        boxResults = self.process_box(b, h, w, threshold)
        if boxResults is None:
            continue
        left, right, top, bot, mess, max_indx, confidence = boxResults
        thick = int((h + w) // 300)
        if self.FLAGS.json:
            resultsForJSON.append(
                {"label": mess, "confidence": float('%.2f' % confidence), "topleft": {"x": left, "y": top},
                 "bottomright": {"x": right, "y": bot}})
            continue

        cv2.rectangle(imgcv,
                      (left, top), (right, bot),
                      self.meta['colors'][max_indx], thick)
        cv2.putText(
            imgcv, mess, (left, top - 12),
            0, 1e-3 * h, self.meta['colors'][max_indx],
               thick // 3)

    if not save: return imgcv

    outfolder = os.path.join(self.FLAGS.imgdir, 'out')
    img_name = os.path.join(outfolder, os.path.basename(im))
    if self.FLAGS.json:
        textJSON = json.dumps(resultsForJSON)
        textFile = os.path.splitext(img_name)[0] + ".json"
        with open(textFile, 'w') as f:
            f.write(textJSON)
        return

    cv2.imwrite(img_name, imgcv)


## ======================================== ##

def parse(self, exclusive=False):
    meta = self.meta
    ext = '.parsed'
    ann = self.FLAGS.annotation
    if not os.path.isdir(ann):
        msg = 'Annotation directory not found {} .'
        exit('Error: {}'.format(msg.format(ann)))
    print('\n{} parsing {}'.format(meta['model'], ann))
    dumps = pascal_voc_clean_xml(ann, meta['labels'], exclusive)
    return dumps


def _batch(self, chunk):
    """
    Takes a chunk of parsed annotations
    returns value for placeholders of net's
    input & loss layer correspond to this chunk
    """
    meta = self.meta
    S, B = meta['side'], meta['num']
    C, labels = meta['classes'], meta['labels']

    # preprocess
    jpg = chunk[0];
    w, h, allobj_ = chunk[1]
    allobj = deepcopy(allobj_)
    path = os.path.join(self.FLAGS.dataset, jpg)
    img = self.preprocess(path, allobj)

    # Calculate regression target
    cellx = 1. * w / S
    celly = 1. * h / S
    for obj in allobj:
        centerx = .5 * (obj[1] + obj[3])  # xmin, xmax
        centery = .5 * (obj[2] + obj[4])  # ymin, ymax
        cx = centerx / cellx
        cy = centery / celly
        if cx >= S or cy >= S: return None, None
        obj[3] = float(obj[3] - obj[1]) / w
        obj[4] = float(obj[4] - obj[2]) / h
        obj[3] = np.sqrt(obj[3])
        obj[4] = np.sqrt(obj[4])
        obj[1] = cx - np.floor(cx)  # centerx
        obj[2] = cy - np.floor(cy)  # centery
        obj += [int(np.floor(cy) * S + np.floor(cx))]

    # show(im, allobj, S, w, h, cellx, celly) # unit test

    # Calculate placeholders' values
    probs = np.zeros([S * S, C])
    confs = np.zeros([S * S, B])
    coord = np.zeros([S * S, B, 4])
    proid = np.zeros([S * S, C])
    prear = np.zeros([S * S, 4])
    for obj in allobj:
        probs[obj[5], :] = [0.] * C
        probs[obj[5], labels.index(obj[0])] = 1.
        proid[obj[5], :] = [1] * C
        coord[obj[5], :, :] = [obj[1:5]] * B
        prear[obj[5], 0] = obj[1] - obj[3] ** 2 * .5 * S  # xleft
        prear[obj[5], 1] = obj[2] - obj[4] ** 2 * .5 * S  # yup
        prear[obj[5], 2] = obj[1] + obj[3] ** 2 * .5 * S  # xright
        prear[obj[5], 3] = obj[2] + obj[4] ** 2 * .5 * S  # ybot
        confs[obj[5], :] = [1.] * B

    # Finalise the placeholders' values
    upleft = np.expand_dims(prear[:, 0:2], 1)
    botright = np.expand_dims(prear[:, 2:4], 1)
    wh = botright - upleft;
    area = wh[:, :, 0] * wh[:, :, 1]
    upleft = np.concatenate([upleft] * B, 1)
    botright = np.concatenate([botright] * B, 1)
    areas = np.concatenate([area] * B, 1)

    # value for placeholder at input layer
    inp_feed_val = img
    # value for placeholder at loss layer
    loss_feed_val = {
        'probs': probs, 'confs': confs,
        'coord': coord, 'proid': proid,
        'areas': areas, 'upleft': upleft,
        'botright': botright
    }

    return inp_feed_val, loss_feed_val


def shuffle(self):
    batch = self.FLAGS.batch
    data = self.parse()
    size = len(data)

    print('Dataset of {} instance(s)'.format(size))
    if batch > size: self.FLAGS.batch = batch = size
    batch_per_epoch = int(size / batch)

    for i in range(self.FLAGS.epoch):
        shuffle_idx = perm(np.arange(size))
        for b in range(batch_per_epoch):
            # yield these
            x_batch = list()
            feed_batch = dict()

            for j in range(b * batch, b * batch + batch):
                train_instance = data[shuffle_idx[j]]
                try:
                    inp, new_feed = self._batch(train_instance)
                except ZeroDivisionError:
                    print("This image's width or height are zeros: ", train_instance[0])
                    print('train_instance:', train_instance)
                    print('Please remove or fix it then try again.')
                    raise

                if inp is None: continue
                x_batch += [np.expand_dims(inp, 0)]

                for key in new_feed:
                    new = new_feed[key]
                    old_feed = feed_batch.get(key, np.zeros((0,) + new.shape))
                    feed_batch[key] = np.concatenate([old_feed, [new]])

            x_batch = np.concatenate(x_batch, 0)
            yield x_batch, feed_batch

        print('Finish {} epoch(es)'.format(i + 1))


## ============================================ ##

# labels20 = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
#    "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]


labels20 = ["Battery", "HDD", "Hub", "Knife", "Lighter",
            "Note Book", "ODD", "Tablet PC", "Scissors", "Smart Phone",
            "Spray", "SSD", "Switch", "USB", "Office Knife",
            "Firearm", "Firearm Parts", "Magazine", "USB_Metal", "SSD_M.2"]


def labels(meta, FLAGS):
    model = os.path.basename(meta['name'])
    if model == 'yolov2_voc':
        print("Model has a VOC model name, loading VOC labels.")
        meta['labels'] = labels20
    else:
        file = FLAGS.labels
        if model == 'yolo':
            print("Model has a coco model name, loading coco labels.")
            file = os.path.join(FLAGS.config, 'coco.names')

        with open(file, 'r') as f:
            meta['labels'] = list()
            labs = [l.strip() for l in f.readlines()]
            for lab in labs:
                if lab == '----': break
                meta['labels'] += [lab]
    if len(meta['labels']) == 0:
        meta['labels'] = labels20


def is_inp(self, name):
    return name.lower().endswith(('.jpg', '.jpeg', '.png'))


def show(im, allobj, S, w, h, cellx, celly):
    for obj in allobj:
        a = obj[5] % S
        b = obj[5] // S
        cx = a + obj[1]
        cy = b + obj[2]
        centerx = cx * cellx
        centery = cy * celly
        ww = obj[3] ** 2 * w
        hh = obj[4] ** 2 * h
        cv2.rectangle(im, (int(centerx - ww / 2), int(centery - hh / 2)),
                      (int(centerx + ww / 2), int(centery + hh / 2)), (0, 0, 255), 2)
    cv2.imshow('result', im)
    cv2.waitKey()
    cv2.destroyAllWindows()


def show2(im, allobj):
    for obj in allobj:
        cv2.rectangle(im, (obj[1], obj[2]), (obj[3], obj[4]), (0, 0, 255), 2)
    cv2.imshow('result', im)
    cv2.waitKey()
    cv2.destroyAllWindows()
