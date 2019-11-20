"""
For the purpose of upgrading the development of Tensorflow AI Vision Project
Edited by Dr. Hyeung-yun Kim


Data preparation
"""
import os
from SSD_settings import *
import numpy as np
import pickle


def cal_iou(box_a, box_b):
    """
    Calculate the Intersection over Union of two boxes
    Each box specified by upper left conner and lower right conner: (x1, y1, x2, y2)

    Return IOU value
    """
    # Calculate intersection, i.e. area of overlap between the 2 boxes
    x_overlap = max(0, min(box_a[2], box_b[2])) - max(box_a[0], box_b[0])
    y_overlap = max(0, min(box_a[3], box_b[3])) - max(box_a[1], box_b[1])
    intersection = x_overlap * y_overlap

    area_box_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_box_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_box_a + area_box_b - intersection
    iou = intersection / union
    return iou


def find_gt_boxes(data_raw, image_file):
    """
    Given feature map sizes, and single training example,
    find all default boxes that exceed IOU overlap thresholds

    Return y_true array that flags the matching default boxes with class ID (-1 means nothing there)
    """
    # Pre-process ground-truth data
    # Convert absolute coordinates to relative coordinates ranging from 0 to 1
    # Read the target class label (note background class label 0, target labels are ints >= 1)
    targets_data = data_raw[image_file]

    targets_class = []
    target_box_coords = [] # relative coordinate
    for target_data in targets_data:
        # find class label
        target_class = target_data['class']
        targets_class.append(target_class)

        # Calculate relative coordinates
        abs_box_coords = target_data['box_coords']
        scale = np.array([IMG_W, IMG_H, IMG_W, IMG_H])
        box_coords = np.array(abs_box_coords) / scale
        target_box_coords.append(box_coords)

    # Initialize y_true to all 0s (0 -> background)
    y_true_len = 0
    for fm_size in FM_SIZES:
        y_true_len += fm_size[0] * fm_size[1] * NUM_DEFAULT_BOXES
    y_true_conf = np.zeros(y_true_len)
    y_true_loc = np.zeros(y_true_len * 4)

    # For each GT box, for each feature map, for each feature map cell, for each default_box:
    # calculate the IOU overlap and annotate the class label
    # count how many box matches we got
    # If we got a math=ch, calculate normalized box coordinates and update y_true_loc
    match_counter = 0
    for i, gt_box_coords in enumerate(target_box_coords):
        y_true_idx = 0
        for fm_size in FM_SIZES:
            fm_h, fm_w = fm_size
            for row in range(fm_h):
                for col in range(fm_w):
                    for db in DEFAULT_BOXES:
                        # Calculate relative box c coordinates for this default box
                        x1_offset, y1_offset, x2_offset, y2_offset = db
                        abs_db_box_coords = np.array([max(0, col+x1_offset),
                                                     max(0, row+y1_offset),
                                                     min(fm_w, col+1+x2_offset),
                                                     min(fm_h, row+1+y2_offset)])
                        scale = np.array([fm_w, fm_h, fm_w, fm_h])
                        db_box_coords = abs_db_box_coords / scale

                        # Calculate IO overlap of GT box and default box
                        iou = cal_iou(gt_box_coords, db_box_coords)

                        # If box matches, i.e. IOU threshold met
                        if iou >= IOU_THRESH:
                            # Update y_true_conf to reflect we found a match, and increment match counter
                            y_true_conf[y_true_idx] = targets_class[i]
                            match_counter += 1

                            # Calculate normalized box coordinates and update y_true_loc
                            abs_box_center = np.array([col+0.5, row+0.5]) # absolute coordinates of center
                            abs_gt_box_coords = gt_box_coords * scale # absolute ground truth box coordinates
                            norm_box_coords = abs_gt_box_coords - np.concatenate((abs_box_center, abs_box_center))
                            y_true_loc[y_true_idx*4 : y_true_idx*4 + 4] = norm_box_coords

                        y_true_idx += 1

    return y_true_conf, y_true_loc, match_counter


def do_data_prep(data_raw):
    """
    Create the y_true array
    data_raw is the dict mapping image_file -> [{'class': class_int, 'box_coords':(x1, y1, x2, y2}, {...}. ...]

    Return a dict {image_file1: {'y_true_conf': y_true_conf, 'y_true_loc':y_true_loc}, image_file2: ...}
    """
    # Prepare the data by populating y_true appropriately
    data_prep = {}
    for image_file in data_raw.keys():
        # Find ground-truth boxes based on IOU overlap
        # populate y_true_conf (class_labels) and y_true_loc (normalized box coordinates)
        y_true_conf, y_true_loc, match_counter = find_gt_boxes(data_raw, image_file)

        # Only want data points where we have matching default boxes
        if match_counter > 0:
            data_prep[image_file] = {'y_true_conf': y_true_conf, 'y_true_loc': y_true_loc}
    return data_prep


if __name__ == '__main__':
    file_dir = "C:\\Users\\mpole\\Dataset\\Xray\\"
    with open(file_dir + 'xray_data_raw_%sx%s.p' % (IMG_W, IMG_H), 'rb') as f:
        data_raw = pickle.load(f)
    data_prep = do_data_prep(data_raw)

    with open(file_dir + 'xray_data_prep_%sx%s.p' % (IMG_W, IMG_H), 'wb') as f:
        pickle.dump(data_prep, f)

    print('Done. Saved prepared data to data_prep_%sx%s.p' % (IMG_W, IMG_H))
    print('Total images with >= 1 matching box: %d' % len(data_prep.keys()))
