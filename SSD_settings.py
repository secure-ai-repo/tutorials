"""
For the purpose of upgrading the development of Tensorflow AI Vision Project
Edited by Dr. Hyeung-yun Kim


Global settings
"""
import tensorflow as tf

# Default boxes
# DEFAULT_BOXES = ((x1_offset, y1_offset, x2_offset, y2_offset), (...), ...)
# Offset is relative to upper-left-corner and lower-right-conner of the feature map cell
DEFAULT_BOXES = ((-0.5, -0.5, 0.5, 0.5), (0.2, 0.2, -0.2, -0.2), (-0.8, -0.2, 0.8, 0.2), (-0.2, -0.8, 0.2, 0.8))
NUM_DEFAULT_BOXES = len(DEFAULT_BOXES)

# Constants
NUM_CLASSES = 21 # 20 objects + 1 background class
NUM_CHANNELS = 3 # grayscale -> 1, RGB -> 3
NUM_PRED_CONF = NUM_DEFAULT_BOXES * NUM_CLASSES # Number of class predictions per feature map cell
NUM_PRED_LOC = NUM_DEFAULT_BOXES * 4 # number of localization regression prediction per feature map cell


# Bounding Box parameters
IOU_THRESH = 0.5 # match ground-truth box to default boxes exceeding this IOU threshold, during data prep
NMS_IOU_THRESH = 0.2 # IOU threshold for non-max suppression

# Negatives-to-positives ratios used to filter training data
NEG_POS_RATIO = 5 # negative:positive = NEG_POS_RATIO:1

# Class confidence threshold to count as detection
CONF_THRESH = 0.9
# Model selection and dependent parameters
MODEL = 'AlexNet' # Alex/Net/VGG16/ResNet50
if MODEL == 'AlexNet':
    # IMG_H, IMG_W = 300, 300
    # FM_SIZES = [[35,36], [17,17], [9,9], [5,5]] # feature map sizes for SSD hooks via Tensorboard visualization (HxW)
    IMG_H, IMG_W = 260, 400
    FM_SIZES = [[31, 48], [15, 23], [8, 12], [4, 6]]
else:
    raise NotImplementedError('Model not implemented')
# Model Hyper-parameters
REG_SCALE = 1e-2 # L2 regularisation strength
LOC_LOSS_WEIGHT = 1. # weight of localization loss: loss = conf_loss + LOC_LOSS_WEIGH * loc_loss

# Training Process
RESUME = False # Resume training from previously saved model?
NUM_EPOCH = 200
BATCH_SIZE = 32 # batch size fot training (relatively small)
VALIDATION_SIZE = 0.05 # fraction of total training set to use as validation set
SAVE_MODEL = True # save trained model to disk
MODEL_SAVE_PATH = './model_SSD.ckpt' # where to save trained model
