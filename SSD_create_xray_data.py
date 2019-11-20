'''
Filename;
Annotation tag;
Upper left corner X;Upper left corner Y;Lower right corner X;Lower right corner Y;
Occluded,On another road;
Origin file;

Create raw data image file
data_raw is a dict mapping image_filename ->
[{'class': class_int, 'box_coords': (x1, y1, x2, y2)}, {...}, ...]

'''

import numpy as np
import pickle
import re
import os
from PIL import Image

# Script config
RESIZE_IMAGE = True  # resize the images and write to 'resized_images/'
GRAYSCALE = False  # convert image to grayscale? this option is only valid if RESIZE_IMAGE==True (FIXME)
TARGET_W, TARGET_H = 400, 260  # 1.74 is weighted avg ratio, 1.65 was for small class image)

###########################
# Execute main script
###########################

# First get mapping from class name string to integer label
# only 20 classes (background class is 0)

classes = ["Battery", "HDD", "Hub", "Knife", "Lighter",
             "Note Book", "ODD", "Tablet PC", "Scissors", "Smart Phone",
             "Spray", "SSD", "Switch", "USB", "Office Knife",
             "Firearm", "Firearm Parts", "Magazine", "USB_Metal", "SSD_M.2"]

merged_annotations = []
label_file_dir = "C:\\Users\\mpole\\Dataset\\Xray\\"
img_file_dir = "C:\\Users\\mpole\\Dataset\\Xray\\img_sample_w400_h260\\"

file_name = os.path.join(label_file_dir + "xray_labels_w400_h260.txt")
with open(file_name, 'r') as f:
    for line in f:
        line = line[:-1]  # strip newline at the end
        integer_label = line.split(',')
        merged_annotations.append(line)

# Create raw data pickle file
data_raw = {}

# Create pickle file to represent dataset
image_files = os.listdir(img_file_dir)
# file_name = "C:\\Users\\mpole\\Dataset\\Xray\\img_sample_w400_h260\\"
for image_file in image_files:
    # Find box coordinates for all signs in this image
    class_list = []
    box_coords_list = []
    for line in merged_annotations:
        if re.search(image_file.split('.')[0], line):
            fields = line.split(' ')

            # Get sign name and assign class label
            class_name = classes[int(fields[3])]
            if class_name not in classes:
                continue  # ignore other names
            _class = fields[3]
            class_list.append(_class)

            # Resize image, get rescaled box coordinates
            box_coords = np.array([float(x) for x in fields[4:8]])

            if RESIZE_IMAGE:
                # Resize the images and write to 'resized_images/'
                # im = Image.open(path + item).convert('RGB')
                image = Image.open(img_file_dir + image_file)
                orig_w, orig_h = image.size

                if GRAYSCALE:
                    image = image.convert('L')  # 8-bit grayscale
                # high-quality downsampling filter vs. Image.ANTIALIAS
                imResize = image.resize((TARGET_W, TARGET_H), Image.LANCZOS)

                resized_dir = img_file_dir + 'resized_images_%dx%d/' % (TARGET_W, TARGET_H)
                if not os.path.exists(resized_dir):
                    os.makedirs(resized_dir)

                imResize.save(os.path.join(resized_dir, image_file), 'PNG')
                # imResize.save(os.path.join(save_dir, rename + str(num)) + '.JPG', 'JPEG', quality=90)

                # Rescale box coordinates
                x_scale = TARGET_W / orig_w
                y_scale = TARGET_H / orig_h

                ulc_x, ulc_y, lrc_x, lrc_y = box_coords
                new_box_coords = (ulc_x * x_scale, ulc_y * y_scale, lrc_x * x_scale, lrc_y * y_scale)
                new_box_coords = [round(x) for x in new_box_coords]
                box_coords = np.array(new_box_coords)

            box_coords_list.append(box_coords)

    if len(class_list) == 0:
        continue  # ignore images with no class_of-interest
    class_list = np.array(class_list)
    box_coords_list = np.array(box_coords_list)

    # Create the list of dicts
    the_list = []
    for i in range(len(box_coords_list)):
        d = {'class': class_list[i], 'box_coords': box_coords_list[i]}
        the_list.append(d)

    data_raw[image_file] = the_list

with open(label_file_dir + 'xray_data_raw_%dx%d.p' % (TARGET_W, TARGET_H), 'wb') as f:
    pickle.dump(data_raw, f)
