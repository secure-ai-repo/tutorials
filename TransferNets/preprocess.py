import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


def preprocess(scopes, inputs):
    from .utils import parse_scopes
    outputs = []
    for scope in scopes:
        model_name = parse_scopes(scope)[0]
        try:
            outputs.append(__preprocess_dict__[model_name](inputs))
        except KeyError:
            found = False
            for (key, fun) in __preprocess_dict__.items():
                if key in model_name.lower():
                    found = True
                    outputs.append(fun(inputs))
                    break
                if not found:
                    outputs.append(inputs)
    if len(outputs) == 1:
        outputs = outputs[0]
    return outputs


def direct(model_name, target_size):
    def _direct(inputs):
        return __preprocess_dict__[model_name](inputs)
    return _direct


def tfslim_preprocess(x):
    x = x.copy()
    x /= 255
    x -= 0.5
    x *= 2.
    return x


# Dictionary for pre-processing functions
__preprocess_dict__ = {
    'inception2': tfslim_preprocess
}
