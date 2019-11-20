import numpy as np
import math
from ..utils.box import BoundBox
from ..utils.nms import NMS


def expit_c(x): # expit
    y= 1/(1+math.exp(-x))
    return y


def max_c(a, b): #MAX
    if(a>b):
        return a
    return b


def box_constructor(meta, net_out_in):  # BOX_CONSTRUCTION
    threshold = meta['thresh']
    tempc, arr_max=0;  sum=0
    anchors = np.asarray(meta['anchors'])
    boxes = list()

    H, W, _ = meta['out_size']
    C = meta['classes']
    B = meta['num']
    
    net_out = net_out_in.reshape([H, W, B, net_out_in.shape[2]/B])
    Classes = net_out[:, :, :, 5:]
    Bbox_pred = net_out[:, :, :, :5]
    probs = np.zeros((H, W, B, C), dtype=np.float32)
    
    for row in range(H):
        for col in range(W):
            for box_loop in range(B):
                arr_max=0
                sum=0;
                Bbox_pred[row, col, box_loop, 4] = expit_c(Bbox_pred[row, col, box_loop, 4])
                Bbox_pred[row, col, box_loop, 0] = (col + expit_c(Bbox_pred[row, col, box_loop, 0])) / W
                Bbox_pred[row, col, box_loop, 1] = (row + expit_c(Bbox_pred[row, col, box_loop, 1])) / H
                Bbox_pred[row, col, box_loop, 2] = math.exp(Bbox_pred[row, col, box_loop, 2]) * anchors[2 * box_loop + 0] / W
                Bbox_pred[row, col, box_loop, 3] = math.exp(Bbox_pred[row, col, box_loop, 3]) * anchors[2 * box_loop + 1] / H

                # SOFTMAX BLOCK, no more pointer juggling
                for class_loop in range(C):
                    arr_max=max_c(arr_max,Classes[row,col,box_loop,class_loop])
                
                for class_loop in range(C):
                    Classes[row,col,box_loop,class_loop]=math.exp(Classes[row,col,box_loop,class_loop]-arr_max)
                    sum+=Classes[row,col,box_loop,class_loop]
                
                for class_loop in range(C):
                    tempc = Classes[row, col, box_loop, class_loop] * Bbox_pred[row, col, box_loop, 4]/sum                    
                    if tempc > threshold:
                        probs[row, col, box_loop, class_loop] = tempc
    
    #NMS
    return NMS(np.ascontiguousarray(probs).reshape(H*W*B,C), np.ascontiguousarray(Bbox_pred).reshape(H*B*W,5))
