"""
For the purpose of upgrading the development of Tensorflow AI Vision Project
Edited by Dr. Hyeung-yun Kim


Tensor operation wrapper classes
"""
import tensorflow.contrib.slim as slim
import tensorflow as tf
import numpy as np

FORM = '{:>6} | {:>6} | {:<32} | {}'
FORM_ = '{}+{}+{}+{}'
LINE = FORM_.format('-' * 7, '-' * 8, '-' * 34, '-' * 15)
HEADER = FORM.format('Source', 'Train?', 'Layer description', 'Output size')


def _shape(tensor):  # work for both tf.Tensor & np.ndarray
    if type(tensor) in [tf.Variable, tf.Tensor]:
        return tensor.get_shape()
    else:
        return tensor.shape


def _name(tensor):
    return tensor.name.split(':')[0]


class BaseOp(object):
    """BaseOp objects initialise with a basecnn's `layer` object and input tensor of that layer `inp`,
    it calculates the output of this layer and place the result in self.out   """
    # let slim take care of the following vars
    _SLIM = ['gamma', 'moving_mean', 'moving_variance']

    def __init__(self, layer, inp, num, roof, feed):  # <= [layer, state, i, roof, self.feed]
        self.inp = inp  # previous layer'
        self.num = num  # int
        self.out = None  # tf.Tensor
        self.lay = layer

        self.scope = '{}-{}'.format(str(self.num), self.lay.type) # name
        self.gap = roof - self.num                                # roof (0) = self.num_layer - self.ntrain (53 - 53)
        self.var = not self.gap > 0
        self.act = 'Load '

        self.convert(feed)
        if self.var:
            self.train_msg = 'Yep! '
        else:
            self.train_msg = 'Nope '
        self.forward()

    def convert(self, feed):
         """convert self.lay to variables & placeholders"""
         for var in self.lay.wshape:
            self.wrap_variable(var)
         for ph in self.lay.h: # ph = 'is_training'
             self.wrap_pholder(ph, feed)

    def wrap_variable(self, var):
        """wrap layer.w into variables"""
        val = self.lay.w.get(var, None)
        if val is None:
            shape = self.lay.wshape[var]
            args = [0., 1e-2, shape]
            if 'moving_mean' in var:
                val = np.zeros(shape)
            elif 'moving_variance' in var:
                val = np.ones(shape)
            else:
                val = np.random.normal(*args)
            self.lay.w[var] = val.astype(np.float32)
            self.act = 'Init '
        if not self.var: return

        val = self.lay.w[var]
        self.lay.w[var] = tf.constant_initializer(val)
        if var in self._SLIM: return
        with tf.variable_scope(self.scope):
            self.lay.w[var] = tf.get_variable(var, shape=self.lay.wshape[var], dtype=tf.float32, initializer=self.lay.w[var])

    def wrap_pholder(self, ph, feed):
        """wrap layer.h into placeholders"""
        phtype = type(self.lay.h[ph])
        if phtype is not dict:
            return
        sig = '{}/{}'.format(self.scope, ph)
        val = self.lay.h[ph]   """layer.h['is_training'] = {'shape': (), 'feed': True, 'default': False} """
        # is_trianing = tf.placeholder_with_default(False, shape=(), name='is_training_')
        self.lay.h[ph] = tf.placeholder_with_default(val['default'], val['shape'], name=sig)
        feed[self.lay.h[ph]] = val['feed'] # feed_dict = {self.lay.h[ph]: True or False} for batch normalization

    def verbalise(self):  # console speaker
        msg = str()
        inp = _name(self.inp.out)
        if inp == 'input': msg = FORM.format('', '', 'input', _shape(self.inp.out)) + '\n'
        if not self.act: return msg
        return msg + FORM.format(self.act, self.train_msg, self.speak(), _shape(self.out))

    def speak(self): pass
    def forward(self): pass


class route(BaseOp):
    def forward(self):
        routes = self.lay.routes
        routes_out = list()
        for r in routes:
            this = self.inp
            while this.lay.number != r:
                this = this.inp
                assert this is not None, 'Routing to non-existence {}'.format(r)
            routes_out += [this.out]
        self.out = tf.concat(routes_out, 3)

    def speak(self):
        msg = 'concat {}'
        return msg.format(self.lay.routes)


class connected(BaseOp):
    def forward(self):
        self.out = tf.nn.xw_plus_b(self.inp.out, self.lay.w['weights'], self.lay.w['biases'], name=self.scope)

    def speak(self):
        layer = self.lay
        args = [layer.inp, layer.out]
        args += [layer.activation]
        msg = 'full {} x {}  {}'
        return msg.format(*args)


class select(connected):
    """a weird connected layer"""
    def speak(self):
        layer = self.lay
        args = [layer.inp, layer.out]
        args += [layer.activation]
        msg = 'sele {} x {}  {}'
        return msg.format(*args)


class extract(connected):
    """a weird connected layer"""
    def speak(self):
        layer = self.lay
        args = [len(layer.inp), len(layer.out)]
        args += [layer.activation]
        msg = 'extr {} x {}  {}'
        return msg.format(*args)


class flatten(BaseOp):
    def forward(self):
        temp = tf.transpose(self.inp.out, [0, 3, 1, 2])
        self.out = slim.flatten(temp, scope=self.scope)

    def speak(self): return 'flat'


class softmax(BaseOp):
    def forward(self):
        self.out = tf.nn.softmax(self.inp.out)

    def speak(self): return 'softmax()'


class avgpool(BaseOp):
    def forward(self):
        self.out = tf.reduce_mean(self.inp.out, [1, 2], name=self.scope)

    def speak(self): return 'avgpool()'


class dropout(BaseOp):
    def forward(self):
        if self.lay.h['pdrop'] is None:
            self.lay.h['pdrop'] = 1.0
        self.out = tf.nn.dropout(self.inp.out, self.lay.h['pdrop'], name=self.scope)

    def speak(self): return 'drop'


class crop(BaseOp):
    def forward(self):
        self.out = self.inp.out * 2. - 1.

    def speak(self):
        return 'scale to (-1, 1)'


class maxpool(BaseOp):
    def forward(self):
        self.out = tf.nn.max_pool2d(self.inp.out, padding='SAME', ksize=[1] + [self.lay.ksize] * 2 + [1],
                                  strides=[1] + [self.lay.stride] * 2 + [1], name=self.scope)

    def speak(self):
        l = self.lay
        return 'maxp {}x{}p{}_{}'.format(l.ksize, l.ksize, l.pad, l.stride)


class leaky(BaseOp):
    def forward(self):
        self.out = tf.maximum(.1 * self.inp.out, self.inp.out, name=self.scope)

    def verbalise(self): pass


class identity(BaseOp):
    def __init__(self, inp):
        self.inp = None
        self.out = inp


class reorg(BaseOp):
    def _forward(self):
        inp = self.inp.out
        shape = inp.get_shape().as_list()
        _, h, w, c = shape
        s = self.lay.stride
        out = list()
        for i in range(int(h / s)):
            row_i = list()
            for j in range(int(w / s)):
                si, sj = s * i, s * j
                boxij = inp[:, si: si + s, sj: sj + s, :]
                flatij = tf.reshape(boxij, [-1, 1, 1, c * s * s])
                row_i += [flatij]
            out += [tf.concat(row_i, 2)]

        self.out = tf.concat(out, 1)

    def forward(self):
        inp = self.inp.out
        s = self.lay.stride #images, ksizes, strides, rates, padding, name
        self.out = tf.extract_image_patches(images=inp,  sizes=[1, s, s, 1],
                                            strides=[1, s, s, 1], rates=[1, 1, 1, 1], padding='VALID')

    def speak(self):
        args = [self.lay.stride] * 2
        msg = 'local flatten {}x{}'
        return msg.format(*args)


class local(BaseOp):
    def forward(self):
        pad = [[self.lay.pad, self.lay.pad]] * 2;
        temp = tf.pad(self.inp.out, [[0, 0]] + pad + [[0, 0]])

        k = self.lay.w['kernels']
        ksz = self.lay.ksize
        half = int(ksz / 2)
        out = list()
        for i in range(self.lay.h_out):
            row_i = list()
            for j in range(self.lay.w_out):
                kij = k[i * self.lay.w_out + j]
                i_, j_ = i + 1 - half, j + 1 - half
                tij = temp[:, i_: i_ + ksz, j_: j_ + ksz, :]
                row_i.append(tf.nn.conv2d(tij, kij, padding='VALID', strides=[1] * 4))
            out += [tf.concat(row_i, 2)]

        self.out = tf.concat(out, 1)

    def speak(self):
        l = self.lay
        args = [l.ksize] * 2 + [l.pad] + [l.stride]
        args += [l.activation]
        msg = 'loca {}x{}p{}_{}  {}'.format(*args)
        return msg


class convolutional(BaseOp):
    def forward(self):
        pad = [[self.lay.pad, self.lay.pad]] * 2;
        temp = tf.pad(self.inp.out, [[0, 0]] + pad + [[0, 0]])  # previous layer's output
        temp = tf.nn.conv2d(temp, self.lay.w['kernel'], padding='VALID',
                            name=self.scope, strides=[1] + [self.lay.stride] * 2 + [1])
        if self.lay.batch_norm:
            if not self.var:
                temp = (temp - self.lay.w['moving_mean'])
                temp /= (np.sqrt(self.lay.w['moving_variance']) + 1e-5)
                temp *= self.lay.w['gamma']
            else:
                args = dict({'center': False,
                             'scale': True,
                             'epsilon': 1e-5,
                             'scope': self.scope,
                             'updates_collections': None,
                             'is_training': self.lay.h['is_training'],
                             'param_initializers': self.lay.w})
                temp = slim.batch_norm(temp, **args)

        self.out = tf.nn.bias_add(temp, self.lay.w['biases'])

    def speak(self):
        l = self.lay
        args = [l.ksize] * 2 + [l.pad] + [l.stride]
        args += [l.batch_norm * '+bnorm']
        args += [l.activation]
        msg = 'conv {}x{}p{}_{}  {}  {}'.format(*args)
        return msg


class conv_select(convolutional):
    def speak(self):
        l = self.lay
        args = [l.ksize] * 2 + [l.pad] + [l.stride]
        args += [l.batch_norm * '+bnorm']
        args += [l.activation]
        msg = 'sele {}x{}p{}_{}  {}  {}'.format(*args)
        return msg


class conv_extract(convolutional):
    def speak(self):
        l = self.lay
        args = [l.ksize] * 2 + [l.pad] + [l.stride]
        args += [l.batch_norm * '+bnorm']
        args += [l.activation]
        msg = 'extr {}x{}p{}_{}  {}  {}'.format(*args)
        return msg


op_types = {
    'convolutional': convolutional,
    'conv-select': conv_select,
    'connected': connected,
    'maxpool': maxpool,
    'leaky': leaky,
    'dropout': dropout,
    'flatten': flatten,
    'avgpool': avgpool,
    'softmax': softmax,
    'identity': identity,
    'crop': crop,
    'local': local,
    'select': select,
    'route': route,
    'reorg': reorg,
    'conv-extract': conv_extract,
    'extract': extract
}

