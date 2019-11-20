"""
Collection of pretrained models
"""
import tensorflow as tf
import warnings

from .utils import init
from .utils import parse_weights
from .utils import parse_scopes
from .utils import pretrained_initializer


def direct(model_name, scope):
    if model_name.startswith('gen'):
        model_name = model_name[3:].lower()
        stem_name = scope.stem.model_name
        try:
            fun = __gen_load_dict__[model_name][stem_name]
        except KeyError:
            fun = load_nothing
    else:
        try:
            fun = __load_dict__[model_name]
        except KeyError:
            fun = load_nothing

    def _direct():
        return fun(scope, return_fn=pretrained_initializer)
    return _direct


def _assign(scopes, values):
    sess = tf.get_default_session()

    scopes = parse_scopes(scopes)
    for scope in scopes:
        sess.run(pretrained_initializer(scope, values))


def load_nothing(scopes, return_fn=_assign):
    return return_fn(scopes, None)


def load_inception2(scopes, return_fn=_assign):
    """ Converted from the TF Slim[2]"""
    filename = 'inception2.npz'
    weights_path = './'
    # get_file(
    #     filename, __model_url__ + 'inception/' + filename,
    #     cache_subdir='models',
    #     file_hash=""
    # )
    values = parse_weights(weights_path)
    return return_fn(scopes, values)


# Dictionary for loading fuctions
__load_dict__ = {
    'inception2': load_inception2
}

__gen_load_dict__ = {
    'yolov2': {
        'darknet19': None #load_ref_yolo_v2_voc
    }
}