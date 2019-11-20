"""
Collection of representative endpoints for each model
"""
from __future__ import absolute_import


def name_inceptions(k, first_block, omit_first=False, pool_last=False, resenet=False):
    names = []
    postfix = ['concat', 'out' if resenet else 'concat']
    for i in range(3):
        first_char = 98 if omit_first is True and i== 0 else 97
        names += ['block%d%c/%s:0' % (i + first_block, j + first_char, postfix[j > 0])
                  for j in range(k[i])]
        if pool_last is True and i < 2:
            names[-1] = 'pool%d/MaxPool:0' % (i + first_block)
    return names


def direct(model_name):
    try:
        return __middle_dict__[model_name]
    except KeyError:
        return ([-1], ['out:0'])


# Dictionary for lists of endpoints
__middle_dict__ = {
    'inception2': (
        [31, 54] + list(range(71, 164, 23)) + list(range(180, 227, 23)),
        name_inceptions[3,5,2],3,
        -4
    )}
