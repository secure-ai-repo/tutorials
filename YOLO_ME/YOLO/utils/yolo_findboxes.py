import numpy as np
from ..utils.box import BoundBox
from ..utils.nms import NMS


def yolo_box_constructor(meta, net_out, threshold):
    sqrt = meta['sqrt'] + 1
    C, B, S = meta['classes'], meta['num'], meta['side']
    SS = S * S  # number of grid cells
    prob_size = SS * C  # class probabilities
    conf_size = SS * B  # confidences for each grid cell

    probs = np.ascontiguousarray(net_out[0: prob_size]).reshape([SS, C])
    confs = np.ascontiguousarray(net_out[prob_size: (prob_size + conf_size)]).reshape([SS, B])
    coords = np.ascontiguousarray(net_out[(prob_size + conf_size):]).reshape([SS, B, 4])
    final_probs = np.zeros([SS, B, C], dtype=np.float32)

    for grid in range(SS):
        for b in range(B):
            coords[grid, b, 0] = (coords[grid, b, 0] + grid % S) / S
            coords[grid, b, 1] = (coords[grid, b, 1] + grid // S) / S
            coords[grid, b, 2] = coords[grid, b, 2] ** sqrt
            coords[grid, b, 3] = coords[grid, b, 3] ** sqrt
            for class_loop in range(C):
                probs[grid, class_loop] = probs[grid, class_loop] * confs[grid, b]

                # print("PROBS",probs[grid,class_loop])
                if probs[grid, class_loop] > threshold:
                    final_probs[grid, b, class_loop] = probs[grid, class_loop]

    return NMS(np.ascontiguousarray(final_probs).reshape(SS * B, C), np.ascontiguousarray(coords).reshape(SS * B, 4))
