"""
For the purpose of upgrading the development of Tensorflow AI Vision Project
Edited by Dr. Hyeung-yun Kim


Run inference using trained model
"""
import tensorflow as tf
from SSD_settings import *
from SSD_model import SSDModel
from SSD_model import nms
import numpy as np
import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt
from optparse import OptionParser
import glob


def run_inference(image, model, sess, target_map):
    """
    Run inference on a given image

    Arguments:
        * image: Numpy array representing a single RGB image
        * model: Dict of tensor references returned by SSD_Model()
        * sess: TensorFlow session reference
    Return
        * Numpy array representing annotated image
    """
    # Save original image in memory
    image = np.array(image)
    image_orig = np.copy(image)

    # Get relevant tensors
    x = model['x']
    is_training = model['is_training']
    preds_conf = model['preds_conf']
    preds_loc = model['preds_loc']
    probs = model['prob']

    # Convert image to PIL Image, resize it, convert to grayscale (if necessary), convert back to numpy array
    image = Image.fromarray(image)
    orig_w, orig_h = image.size
    image = image.resize((IMG_W, IMG_H), Image.LANCZOS)  # high-quality downsampling filter
    image = np.asarray(image)

    images = np.array([image])  # create a 'batch' of 1 image

    # Performing object detection
    preds_conf_val, preds_loc_val, probs_val = sess.run([preds_conf, preds_loc, probs], feed_dict={x: images, is_training: False})

    # Gather class predictions and confidence values
    y_pred_conf = preds_conf_val[0] # batch size of 1, so just take [0]
    y_pred_conf = y_pred_conf.astype('float32')
    prob = probs_val[0]

    # Gather localization predictions
    y_pred_loc = preds_loc_val[0]

    boxes = nms(y_pred_conf, y_pred_loc, prob)

    # Rescale boxes' coordinates back to original image's dimension
    # Recall boxes = [{x1, y1, x2, y2, cls, cls_prob}, {...}, ...]
    scale = np.array([orig_w/IMG_W, orig_h/IMG_H, orig_w/IMG_W, orig_h/IMG_H])
    if len(boxes) > 0:
        boxes[:, :4] = boxes[:, :4] * scale

    # Draw and annotate boxes over original image, and return annotated image
    image = image_orig
    for box in boxes:
        # Get box parameters
        box_coords = [int(round(x)) for x in box[:4]]
        cls = int(box[4])
        cls_prob = box[5]
        # Annotate image

        image = cv2.rectangle(image, tuple(box_coords[:2]), tuple(box_coords[2:]), (0,255,0))
        label_str = '%s %.2f' % (target_map[cls], cls_prob)
        image = cv2.putText(image, label_str, (box_coords[0], box_coords[1]), 0, 0.5, (0,255,0), 1, cv2.LINE_AA)

    return image


def generate_output(input_files):
    """
    Generate annotated images
    """
    # First, load mapping from integer class ID to target name string
    target_map = {0: "Battery", 1: "HDD", 2: "Hub", 3: "Knife", 4: "Lighter", 5: "Note Book", 6: "ODD", 7: "Tablet PC",
                  8: "Scissors", 9: "Smart Phone", 10: "Spray", 11: "SSD", 12: "Switch", 13: "USB", 14: "Office Knife",
                  15: "Firearm", 16: "Firearm Parts", 17: "Magazine", 18: "USB_Metal", 19: "SSD_M.2", 20: "background"}

    # create output directory 'inference_out/' if needed
    if not os.path.isdir('C:\\Users\\mpole\\Dataset\\Xray\\inference_out\\'):
        try:
            os.mkdir('C:\\Users\\mpole\\Dataset\\Xray\\inference_out\\')
        except FileExistsError:
            print('Error: can not create ./inference_out')
            return

    # Launch the graph
    with tf.Graph().as_default(), tf.Session() as sess:
        # 'Instantiate neural network, get relevant tensors
        model = SSDModel()

        # Load trained model
        saver = tf.train.Saver()
        print('Restoring previously trained model at %s' % MODEL_SAVE_PATH)
        saver.restore(sess, MODEL_SAVE_PATH)

        for image_file in input_files:
            print('Running infernece on %s' % image_file)
            image_orig = np.asarray(Image.open(image_file))
            image = run_inference(image_orig, model, sess, target_map)

            head, tail = os.path.split(image_file)
            plt.imsave('C:\\Users\\mpole\\Dataset\\Xray\\inference_out\\%s' % tail, image)
        print('Output saved in inference_out/')


if __name__=='__main__':
    # parser = OptionParser()
    # parser.add_option('-i', '--input_dir', dest='input_dir', help=' ')

    # Get and parse command line options
    # options, arg = parser.parse_args()
    # input_dir = options.input_dir

    input_dir = "C:\\Users\\mpole\\Dataset\\Xray\\ssd_input_images\\"
    input_files = glob.glob(input_dir + '*.*')

    generate_output(input_files)