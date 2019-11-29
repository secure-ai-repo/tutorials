"""
For the purpose of upgrading the development of Tensorflow AI Vision Project
Edited by Dr. Hyeung-yun Kim


YOLO network model
"""
import warnings
import time
import os
import numpy as np
import tensorflow as tf
import pickle
from os.path import basename


def model_name(file_path):
    file_name = basename(file_path)
    ext = str()
    if '.' in file_name:  # exclude extension
        file_name = file_name.split('.')
        ext = file_name[-1]
        file_name = '.'.join(file_name[:-1])
    if ext == str() or ext == 'meta':  # ckpt file
        file_name = file_name.split('-')
        num = int(file_name[-1])
        return '-'.join(file_name[:-1])
    if ext == 'weights':
        return file_name


def parser(model):
    """
    Read the .cfg file to extract layers into `layers`
    as well as model-specific parameters into `meta`
    """

    def _parse(l, i=1):
        return l.split('=')[i].strip()

    with open(model, 'rb') as f:
        lines = f.readlines()

    lines = [line.decode() for line in lines]

    meta = dict();
    layers = list()
    h, w, c = [int()] * 3
    layer = dict()
    for line in lines:
        line = line.strip()
        line = line.split('#')[0]
        if '[' in line:
            if layer != dict():
                if layer['type'] == '[net]':
                    h = layer['height']
                    w = layer['width']
                    c = layer['channels']
                    meta['net'] = layer
                else:
                    if layer['type'] == '[crop]':
                        h = layer['crop_height']
                        w = layer['crop_width']
                    layers += [layer]
            layer = {'type': line}
        else:
            try:
                i = float(_parse(line))
                if i == int(i): i = int(i)
                layer[line.split('=')[0].strip()] = i
            except:
                try:
                    key = _parse(line, 0)
                    val = _parse(line, 1)
                    layer[key] = val
                except:
                    '~~~~~~'

    meta.update(layer)  # last layer contains meta info
    if 'anchors' in meta:
        splits = meta['anchors'].split(',')
        anchors = [float(x.strip()) for x in splits]
        meta['anchors'] = anchors
    meta['model'] = model  # path to cfg, not model name
    meta['inp_size'] = [h, w, c]
    return layers, meta


def cfg_yielder(model_cfg, binary):
    """
    yielding each layer information to initialize `layer`
    """
    layers, meta = parser(model_cfg)
    yield meta
    h, w, c = meta['inp_size']
    l = w * h * c

    # Start yielding
    flat = False  # flag for 1st dense layer
    conv = '.conv.' in model_cfg
    for i, d in enumerate(layers):
        # -----------------------------------------------------
        if d['type'] == '[crop]':
            yield ['crop', i]
        # -----------------------------------------------------
        elif d['type'] == '[local]':
            n = d.get('filters', 1)
            size = d.get('size', 1)
            stride = d.get('stride', 1)
            pad = d.get('pad', 0)
            activation = d.get('activation', 'logistic')
            w_ = (w - 1 - (1 - pad) * (size - 1)) // stride + 1
            h_ = (h - 1 - (1 - pad) * (size - 1)) // stride + 1
            yield ['local', i, size, c, n, stride,
                   pad, w_, h_, activation]
            if activation != 'linear': yield [activation, i]
            w, h, c = w_, h_, n
            l = w * h * c
        # -----------------------------------------------------
        elif d['type'] == '[convolutional]':
            n = d.get('filters', 1)
            size = d.get('size', 1)
            stride = d.get('stride', 1)
            pad = d.get('pad', 0)
            padding = d.get('padding', 0)
            if pad: padding = size // 2
            activation = d.get('activation', 'logistic')
            batch_norm = d.get('batch_normalize', 0) or conv
            yield ['convolutional', i, size, c, n, stride, padding, batch_norm, activation]
            if activation != 'linear': yield [activation, i]
            w_ = (w + 2 * padding - size) // stride + 1
            h_ = (h + 2 * padding - size) // stride + 1
            w, h, c = w_, h_, n
            l = w * h * c
        # -----------------------------------------------------
        elif d['type'] == '[maxpool]':
            stride = d.get('stride', 1)
            size = d.get('size', stride)
            padding = d.get('padding', (size - 1) // 2)
            yield ['maxpool', i, size, stride, padding]
            w_ = (w + 2 * padding) // d['stride']
            h_ = (h + 2 * padding) // d['stride']
            w, h = w_, h_
            l = w * h * c
        # -----------------------------------------------------
        elif d['type'] == '[avgpool]':
            flat = True;
            l = c
            yield ['avgpool', i]
        # -----------------------------------------------------
        elif d['type'] == '[softmax]':
            yield ['softmax', i, d['groups']]
        # -----------------------------------------------------
        elif d['type'] == '[connected]':
            if not flat:
                yield ['flatten', i]
                flat = True
            activation = d.get('activation', 'logistic')
            yield ['connected', i, l, d['output'], activation]
            if activation != 'linear': yield [activation, i]
            l = d['output']
        # -----------------------------------------------------
        elif d['type'] == '[dropout]':
            yield ['dropout', i, d['probability']]
        # -----------------------------------------------------
        elif d['type'] == '[select]':
            if not flat:
                yield ['flatten', i]
                flat = True
            inp = d.get('input', None)
            if type(inp) is str:
                file = inp.split(',')[0]
                layer_num = int(inp.split(',')[1])
                with open(file, 'rb') as f:
                    profiles = pickle.load(f, encoding='latin1')[0]
                layer = profiles[layer_num]
            else:
                layer = inp
            activation = d.get('activation', 'logistic')
            d['keep'] = d['keep'].split('/')
            classes = int(d['keep'][-1])
            keep = [int(c) for c in d['keep'][0].split(',')]
            keep_n = len(keep)
            train_from = classes * d['bins']
            for count in range(d['bins'] - 1):
                for num in keep[-keep_n:]:
                    keep += [num + classes]
            k = 1
            while layers[i - k]['type'] not in ['[connected]', '[extract]']:
                k += 1
                if i - k < 0:
                    break
            if i - k < 0:
                l_ = l
            elif layers[i - k]['type'] == 'connected':
                l_ = layers[i - k]['output']
            else:
                l_ = layers[i - k].get('old', [l])[-1]
            yield ['select', i, l_, d['old_output'], activation, layer, d['output'], keep, train_from]
            if activation != 'linear': yield [activation, i]
            l = d['output']
        # -----------------------------------------------------
        elif d['type'] == '[conv-select]':
            n = d.get('filters', 1)
            size = d.get('size', 1)
            stride = d.get('stride', 1)
            pad = d.get('pad', 0)
            padding = d.get('padding', 0)
            if pad: padding = size // 2
            activation = d.get('activation', 'logistic')
            batch_norm = d.get('batch_normalize', 0) or conv
            d['keep'] = d['keep'].split('/')
            classes = int(d['keep'][-1])
            keep = [int(x) for x in d['keep'][0].split(',')]

            segment = classes + 5
            assert n % segment == 0, 'conv-select: segment failed'
            bins = n // segment
            keep_idx = list()
            for j in range(bins):
                offset = j * segment
                for k in range(5):
                    keep_idx += [offset + k]
                for k in keep:
                    keep_idx += [offset + 5 + k]
            w_ = (w + 2 * padding - size) // stride + 1
            h_ = (h + 2 * padding - size) // stride + 1
            c_ = len(keep_idx)
            yield ['conv-select', i, size, c, n, stride, padding, batch_norm, activation, keep_idx, c_]
            w, h, c = w_, h_, c_
            l = w * h * c
        # -----------------------------------------------------
        elif d['type'] == '[conv-extract]':
            file = d['profile']
            with open(file, 'rb') as f:
                profiles = pickle.load(f, encoding='latin1')[0]
            inp_layer = None
            inp = d['input']
            out = d['output']
            inp_layer = None
            if inp >= 0:
                inp_layer = profiles[inp]
            if inp_layer is not None:
                assert len(inp_layer) == c, 'Conv-extract does not match input dimension'
            out_layer = profiles[out]

            n = d.get('filters', 1)
            size = d.get('size', 1)
            stride = d.get('stride', 1)
            pad = d.get('pad', 0)
            padding = d.get('padding', 0)
            if pad: padding = size // 2
            activation = d.get('activation', 'logistic')
            batch_norm = d.get('batch_normalize', 0) or conv

            k = 1
            find = ['[convolutional]', '[conv-extract]']
            while layers[i - k]['type'] not in find:
                k += 1
                if i - k < 0: break
            if i - k >= 0:
                previous_layer = layers[i - k]
                c_ = previous_layer['filters']
            else:
                c_ = c

            yield ['conv-extract', i, size, c_, n, stride, padding, batch_norm, activation, inp_layer, out_layer]
            if activation != 'linear': yield [activation, i]
            w_ = (w + 2 * padding - size) // stride + 1
            h_ = (h + 2 * padding - size) // stride + 1
            w, h, c = w_, h_, len(out_layer)
            l = w * h * c
        # -----------------------------------------------------
        elif d['type'] == '[extract]':
            if not flat:
                yield ['flatten', i]
                flat = True
            activation = d.get('activation', 'logistic')
            file = d['profile']
            with open(file, 'rb') as f:
                profiles = pickle.load(f, encoding='latin1')[0]
            inp_layer = None
            inp = d['input']
            out = d['output']
            if inp >= 0:
                inp_layer = profiles[inp]
            out_layer = profiles[out]
            old = d['old']
            old = [int(x) for x in old.split(',')]
            if inp_layer is not None:
                if len(old) > 2:
                    h_, w_, c_, n_ = old
                    new_inp = list()
                    for p in range(c_):
                        for q in range(h_):
                            for r in range(w_):
                                if p not in inp_layer:
                                    continue
                                new_inp += [r + w * (q + h * p)]
                    inp_layer = new_inp
                    old = [h_ * w_ * c_, n_]
                assert len(inp_layer) == l, 'Extract does not match input dimension'
            d['old'] = old
            yield ['extract', i] + old + [activation] + [inp_layer, out_layer]
            if activation != 'linear': yield [activation, i]
            l = len(out_layer)
        # -----------------------------------------------------
        elif d['type'] == '[route]':  # add new layer here
            routes = d['layers']
            if type(routes) is int:
                routes = [routes]
            else:
                routes = [int(x.strip()) for x in routes.split(',')]
            routes = [i + x if x < 0 else x for x in routes]
            for j, x in enumerate(routes):
                lx = layers[x];
                xtype = lx['type']
                _size = lx['_size'][:3]
                if j == 0:
                    h, w, c = _size
                else:
                    h_, w_, c_ = _size
                    assert w_ == w and h_ == h, 'Routing incompatible conv sizes'
                    c += c_
            yield ['route', i, routes]
            l = w * h * c
        # -----------------------------------------------------
        elif d['type'] == '[reorg]':
            stride = d.get('stride', 1)
            yield ['reorg', i, stride]
            w = w // stride;
            h = h // stride;
            c = c * (stride ** 2)
            l = w * h * c
        # -----------------------------------------------------
        else:
            exit('Layer {} not implemented'.format(d['type']))

        d['_size'] = list([h, w, c, l, flat])

    if not flat:
        meta['out_size'] = [h, w, c]
    else:
        meta['out_size'] = l


## ========================================================
class Layer(object):
    def __init__(self, *args):
        self._signature = list(args)
        self.type = list(args)[0]
        self.number = list(args)[1]

        self.w = dict()  # weights
        self.h = dict()  # placeholders
        self.wshape = dict()  # weight shape
        self.wsize = dict()  # weight size
        self.setup(*args[2:])  # set attr up
        self.present()
        for var in self.wshape:
            shp = self.wshape[var]
            size = np.prod(shp)
            self.wsize[var] = size

    def load(self, src_loader):
        var_lay = src_loader.VAR_LAYER
        if self.type not in var_lay:
            return
        wdict = self.load_weights(src_loader)
        if wdict is not None:
            self.recollect(wdict)

    def load_weights(self, src_loader):
        val = src_loader([self.presenter])
        if val is None:
            return None
        else:
            return val.w

    @property
    def signature(self):
        return self._signature

    # For comparing two layers
    def __eq__(self, other):
        return self.signature == other.signature

    def __ne__(self, other):
        return not self.__eq__(other)

    def varsig(self, var):
        if var not in self.wshape:
            return None
        sig = str(self.number)
        sig += '-' + self.type
        sig += '/' + var
        return sig

    def recollect(self, w):
        self.w = w

    def present(self):
        self.presenter = self

    def setup(self, *args):
        pass

    def finalize(self):
        pass


class avgpool_layer(Layer):
    pass


class crop_layer(Layer):
    pass


class maxpool_layer(Layer):
    def setup(self, ksize, stride, pad):
        self.stride = stride
        self.ksize = ksize
        self.pad = pad


class softmax_layer(Layer):
    def setup(self, groups):
        self.groups = groups


class dropout_layer(Layer):
    def setup(self, p):
        self.h['pdrop'] = dict({
            'feed': p,  # for training
            'dfault': 1.0,  # for testing
            'shape': ()
        })


class route_layer(Layer):
    def setup(self, routes):
        self.routes = routes


class reorg_layer(Layer):
    def setup(self, stride):
        self.stride = stride


class local_layer(Layer):
    def setup(self, ksize, c, n, stride, pad, w_, h_, activation):
        self.pad = pad * int(ksize / 2)
        self.activation = activation
        self.stride = stride
        self.ksize = ksize
        self.h_out = h_
        self.w_out = w_

        self.dnshape = [h_ * w_, n, c, ksize, ksize]
        self.wshape = dict({'biases': [h_ * w_ * n], 'kernels': [h_ * w_, ksize, ksize, c, n]})

    def finalize(self, _):
        weights = self.w['kernels']
        if weights is None:
            return
        weights = weights.reshape(self.dnshape)
        weights = weights.transpose([0, 3, 4, 2, 1])
        self.w['kernels'] = weights


class conv_extract_layer(Layer):
    def setup(self, ksize, c, n, stride, pad, batch_norm, activation, inp, out):
        if inp is None: inp = range(c)
        self.activation = activation
        self.batch_norm = batch_norm
        self.stride = stride
        self.ksize = ksize
        self.pad = pad
        self.inp = inp
        self.out = out
        self.wshape = dict({'biases': [len(out)], 'kernel': [ksize, ksize, len(inp), len(out)]})

    @property
    def signature(self):
        sig = ['convolutional']
        sig += self._signature[1:-2]
        return sig

    def present(self):
        args = self.signature
        self.presenter = convolutional_layer(*args)

    def recollect(self, w):
        if w is None:
            self.w = w
            return
        k = w['kernel']
        b = w['biases']
        k = np.take(k, self.inp, 2)
        k = np.take(k, self.out, 3)
        b = np.take(b, self.out)
        assert1 = k.shape == tuple(self.wshape['kernel'])
        assert2 = b.shape == tuple(self.wshape['biases'])
        assert assert1 and assert2, 'Dimension not matching in {} recollect'.format(self._signature)
        self.w['kernel'] = k
        self.w['biases'] = b


class conv_select_layer(Layer):
    def setup(self, ksize, c, n, stride, pad, batch_norm, activation, keep_idx, real_n):
        self.batch_norm = bool(batch_norm)
        self.activation = activation
        self.keep_idx = keep_idx
        self.stride = stride
        self.ksize = ksize
        self.pad = pad
        self.wshape = dict({'biases': [real_n], 'kernel': [ksize, ksize, c, real_n]})
        if self.batch_norm:
            self.wshape.update({'moving_variance': [real_n], 'moving_mean': [real_n], 'gamma': [real_n]})
            self.h['is_training'] = {'shape': (), 'feed': True, 'default': False}

    @property
    def signature(self):
        sig = ['convolutional']
        sig += self._signature[1:-2]
        return sig

    def present(self):
        args = self.signature
        self.presenter = convolutional_layer(*args)

    def recollect(self, w):
        if w is None:
            self.w = w
            return
        idx = self.keep_idx
        k = w['kernel']
        b = w['biases']
        self.w['kernel'] = np.take(k, idx, 3)
        self.w['biases'] = np.take(b, idx)
        if self.batch_norm:
            m = w['moving_mean']
            v = w['moving_variance']
            g = w['gamma']
            self.w['moving_mean'] = np.take(m, idx)
            self.w['moving_variance'] = np.take(v, idx)
            self.w['gamma'] = np.take(g, idx)


class convolutional_layer(Layer):
    def setup(self, ksize, c, n, stride, pad, batch_norm, activation):
        self.batch_norm = bool(batch_norm)
        self.activation = activation
        self.stride = stride
        self.ksize = ksize
        self.pad = pad
        self.dnshape = [n, c, ksize, ksize]  # darknet shape
        self.wshape = dict({'biases': [n], 'kernel': [ksize, ksize, c, n]})
        if self.batch_norm:
            self.wshape.update({'moving_variance': [n], 'moving_mean': [n], 'gamma': [n]})
            self.h['is_training'] = {'feed': True, 'default': False, 'shape': ()}

    def finalize(self, _):
        """deal with darknet"""
        kernel = self.w['kernel']
        if kernel is None: return
        kernel = kernel.reshape(self.dnshape)
        kernel = kernel.transpose([2, 3, 1, 0])
        self.w['kernel'] = kernel


class extract_layer(Layer):
    def setup(self, old_inp, old_out, activation, inp, out):
        if inp is None: inp = range(old_inp)
        self.activation = activation
        self.old_inp = old_inp
        self.old_out = old_out
        self.inp = inp
        self.out = out
        self.wshape = {'biases': [len(self.out)], 'weights': [len(self.inp), len(self.out)]}

    @property
    def signature(self):
        sig = ['connected']
        sig += self._signature[1:-2]
        return sig

    def present(self):
        args = self.signature
        self.presenter = connected_layer(*args)

    def recollect(self, val):
        w = val['weights']
        b = val['biases']
        if w is None: self.w = val; return
        w = np.take(w, self.inp, 0)
        w = np.take(w, self.out, 1)
        b = np.take(b, self.out)
        assert1 = w.shape == tuple(self.wshape['weights'])
        assert2 = b.shape == tuple(self.wshape['biases'])
        assert assert1 and assert2, 'Dimension does not match in {} recollect'.format(self._signature)

        self.w['weights'] = w
        self.w['biases'] = b


class select_layer(Layer):
    def setup(self, inp, old, activation, inp_idx, out, keep, train):
        self.old = old
        self.keep = keep
        self.train = train
        self.inp_idx = inp_idx
        self.activation = activation
        inp_dim = inp
        if inp_idx is not None:
            inp_dim = len(inp_idx)
        self.inp = inp_dim
        self.out = out
        self.wshape = {'biases': [out], 'weights': [inp_dim, out]}

    @property
    def signature(self):
        sig = ['connected']
        sig += self._signature[1:-4]
        return sig

    def present(self):
        args = self.signature
        self.presenter = connected_layer(*args)

    def recollect(self, val):
        w = val['weights']
        b = val['biases']
        if w is None: self.w = val; return
        if self.inp_idx is not None:
            w = np.take(w, self.inp_idx, 0)

        keep_b = np.take(b, self.keep)
        keep_w = np.take(w, self.keep, 1)
        train_b = b[self.train:]
        train_w = w[:, self.train:]
        self.w['biases'] = np.concatenate((keep_b, train_b), axis=0)
        self.w['weights'] = np.concatenate((keep_w, train_w), axis=1)


class connected_layer(Layer):
    def setup(self, input_size, output_size, activation):
        self.activation = activation
        self.inp = input_size
        self.out = output_size
        self.wshape = {'biases': [self.out], 'weights': [self.inp, self.out]
                       }

    def finalize(self, transpose):
        weights = self.w['weights']
        if weights is None: return
        shp = self.wshape['weights']
        if not transpose:
            weights = weights.reshape(shp[::-1])
            weights = weights.transpose([1, 0])
        else:
            weights = weights.reshape(shp)
        self.w['weights'] = weights


darkops = {
    'dropout': dropout_layer,
    'connected': connected_layer,
    'maxpool': maxpool_layer,
    'convolutional': convolutional_layer,
    'avgpool': avgpool_layer,
    'softmax': softmax_layer,
    'crop': crop_layer,
    'local': local_layer,
    'select': select_layer,
    'route': route_layer,
    'reorg': reorg_layer,
    'conv-select': conv_select_layer,
    'conv-extract': conv_extract_layer,
    'extract': extract_layer
}


def create_layerop(ltype, num, *args):
    op_class = darkops.get(ltype, Layer)
    return op_class(ltype, num, *args)


class loader(object):
    """ interface to work with both .weights and .ckpt files in loading / recollecting / resolving mode"""
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

    def load(self, *args): pass


class weights_loader(loader):
    """one who understands .weights files order of param flattened into .weights file"""
    _W_ORDER = dict({'convolutional': ['biases', 'gamma', 'moving_mean', 'moving_variance', 'kernel'],
                     'connected': ['biases', 'weights'],
                     'local': ['biases', 'kernels']})

    def load(self, path, src_layers):
        self.src_layers = src_layers
        walker = weights_walker(path)

        for i, layer in enumerate(src_layers):
            if layer.type not in self.VAR_LAYER:
                continue
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
        self.eof = False
        self.path = path
        if path is None:
            self.eof = True
            return
        else:
            self.size = os.path.getsize(path)  # save the path
            major, minor, revision, seen = np.memmap(path, shape=(), mode='r', offset=0, dtype='({})i4,'.format(4))
            self.transpose = major > 1000 or minor > 1000
            self.offset = 16 + 4  # check weight data

    def walk(self, size):
        if self.eof: return None
        end_point = self.offset + 4 * size
        assert end_point <= self.size, 'Over-read {}'.format(self.path)

        float32_1D_array=np.memmap(self.path, shape=(), mode='r', offset=self.offset, dtype='({})float32,'.format(size))

        self.offset = end_point
        if end_point == self.size:
            self.eof = True
        return float32_1D_array


class BaseCNN(object):
    _EXT = '.weights'

    def __init__(self, FLAGS):
        # self.get_weight_src(FLAGS)
        self.src_bin = FLAGS.model + self._EXT
        self.src_bin = FLAGS.binary + self.src_bin
        self.src_bin = os.path.abspath(self.src_bin)
        exist = os.path.isfile(self.src_bin)

        if FLAGS.load == str(): FLAGS.load = int()
        if type(FLAGS.load) is int:
            self.src_cfg = FLAGS.model
            if FLAGS.load:
                self.src_bin = None
            elif not exist:
                self.src_bin = None
        else:
            self.src_bin = FLAGS.load
            name = model_name(FLAGS.load)
            self.src_cfg = os.path.join(FLAGS.config, name + '.cfg')
            if not os.path.isfile(self.src_cfg):
                warnings.warn('{} not found, use {} instead'.format(self.src_cfg, FLAGS.model))
                self.src_cfg = FLAGS.model
            FLAGS.load = int()  # None

        self.modify = False
        print('Parsing {}'.format(FLAGS.model))
        args = [FLAGS.model, FLAGS.binary]
        cfg_layers = cfg_yielder(*args)
        self.meta = dict();
        self.layers = list()
        for i, info in enumerate(cfg_layers):
            if i == 0:
                self.meta = info;
                continue
            else:
                new = create_layerop(*info)
            self.layers.append(new)

        print('Loading {} ...'.format(self.src_bin))
        start = time.time()
        wgts_loader = weights_loader(self.src_bin, self.layers)

        for layer in self.layers:
            var_lay = wgts_loader.VAR_LAYER
            if layer.type not in var_lay: return

            val = wgts_loader([layer.presenter])
            if val is None: wdict =None
            else: wdict = val.w

            if wdict is not None:
                layer.w = wdict

        stop = time.time()
        print('Finished in {}s'.format(stop - start))
