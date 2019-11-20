from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import warnings

from contextlib import contextmanager
from distutils.version import LooseVersion

from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib.layers.python.layers.utils import collect_named_outputs
from tensorflow.python.framework import ops

from .layers import conv2d

try:
    import cv2
except:
    cv2 = None

__middles__ = 'middles'
__outputs__ = 'outputs'


def parse_scopes(inputs):
    if not isinstance(inputs, list):
        inputs = [inputs]
    outputs = []
    for scope_or_tensor in inputs:
        if isinstance(scope_or_tensor, tf.Tensor):
            outputs.append(scope_or_tensor.aliases[0])
        elif isinstance(scope_or_tensor, str):
            outputs.append(scope_or_tensor)
        else:
            outputs.append(None)
    return outputs


def get_bottleneck(scope=None):
    scope = parse_scopes(scope)[0]
    return tf.get_collection(__middles__, scope=scope + '/')[-1]


def get_middles(scope=None):
    scope = parse_scopes(scope)[0]
    return tf.get_collection(__middles__, scope=scope + '/')


def get_outputs(scope=None):
    scope = parse_scopes(scope)[0]
    return tf.get_collection(__outputs__, scope=scope + '/')


def get_weights(scope=None):
    scope = parse_scopes(scope)[0]
    return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope + '/')


def init(scopes):
    sess = tf.get_default_session()
    if not isinstance(scopes, list):
        scopes = [scopes]
    for scope in scopes:
        sess.run(tf.variables_initializer(get_weights(scope)))


def var_scope(name):
    def decorator(func):
        def wrapper(*args, **kwargs):
            stem = kwargs.get('stem', False)
            scope = kwargs.get('scope', name)
            reuse = kwargs.get('resue', None)
            with tf.variable_scope(scope, reuse=reuse):
                x = func(*args, **kwargs)
                if func.__name__ == 'wrapper':
                    from .middles import direct as p0
                    from .preprocess import direct as p1
                    from .pretrained import direct as p2
                    _scope = tf.get_variable_scope().name
                    _name = tf.get_default_graph().get_name_scope()
                    _input_shape = tuple([i.value for i in args[0].shape[1:3]])
                    _outs = get_outputs(_name)
                    for i in p0(name)[0]:
                        collect_named_outputs(__middles__, _scope, _outs[i])
                    if stem:
                        x.aliases.insert(0, _scope)
                        x.p = get_middles(_name)[p0(name)[2]]
                    else:
                        x.logits = get_outputs(_name)[-2]
                    setattr(x, 'preprocess', p1(name, _input_shape))
                    setattr(x, 'pretrained', p2(name, x))
                    setattr(x, 'get_bottleneck', lambda: get_bottleneck(_scope))
                    setattr(x, 'get_middles', lambda: get_middles(_name))
                    setattr(x, 'get_middles', lambda: get_outputs(_name))
                    setattr(x, 'get_middles', lambda: get_weights(_name))
                return x
        return wrapper
    return decorator


def ops_to_outputs(func):
    def wrapper(*args, **kwargs):
        x = func(*args, **kwargs)
        x = collect_named_outputs(__outputs__, tf.get_variable_scope().name, x)
    return wrapper


@contextmanager
def arg_scopes(l):
    for x in l:
        x.__enter__()
    yield


def set_args(largs, conv_bias=True):
    def real_set_args(func):
        def wrapper(*args, **kwargs):
            is_training = kwargs.get('is_training', False)
            layers = sum([x for (x, y) in largs(is_training)], [])
            layers_args = [arg_scope(x, **y) for(x, y) in largs(is_training)]
            if not conv_bias:
                layers_args += [arg_scope([conv2d], biases_initializer=None)]
            with arg_scope(layers, outputs_collections=__outputs__):
                with arg_scopes(layers_args):
                    x = func(*args, **kwargs)
                    x.model_name = func.__name__
                    return x
        return wrapper
    return real_set_args


def pretrained_initializer(scope, values):
    weights = get_weights(scope)

    if values is None:
        return tf.variables_initializer()

    # excluding weights in Optimizer
    if len(weights) > len(values):
        weights = weights[:len(values)]

    if len(weights) != len(values):
        values = values[:len(weights)]

    if scope.dtype == tf.float16:
        ops = [weights[0].assign(np.asarray(values[0], dtype=np.float16))]
        for (w, v) in zip(weights[1:-2], values[1:-2]):
            w.load(np.asarray(v, dtype=np.float16))
        if weights[-1].shape != values[-1].shape:
            ops += [w.initializer for w in weights[-2:]]
        else:
            for (w, v) in zip(weights[-2:], values[-2:]):
                w.load(np.asarray(v, dtype=np.float16))
        return ops

    ops = [w.assign(v) for (w, v) in zip(weights[:-2], values[:-2])]
    if weights[-1].shape != values[-1].shape:
        ops += [w.initializer for w in weights[-2:]]
    else:
        if weights[-2].shape != values[-2].shape:
            values[-2] = values[-2].reshape(weights[-2].shape)
        ops += [w.assign(v) for (w, v) in zip(weights[-2:], values[-2:])]
    return ops


def parse_weights(weight_path, move_rules=None):
    data = np.load(weight_path, encoding='byte', allow_pickle=True)
    values = data['values']
    return values

