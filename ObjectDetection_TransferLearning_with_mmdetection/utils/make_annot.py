import os
import pandas as pd
import json
import glob
import numpy as np
import pickle
import cv2
import mmcv
import random
from scipy import io

def annot(images_path, label_dict, train_data=True):
    """Make annotations for mmdetection library from Open Image Dataset Source.
    Args:
        images_path (str): path to images.
        label_dict (dict (str: int)): dictionary with matching class names to class labels.
        train_data (boolean): if True - train data, else: validation data.
    """
    label_path = images_path+'Label/*.txt'
    save_val_path = './annotations_val.pkl'
    save_train_path = './annotations_train.pkl'
    annotations_train = []
    annotations_val = []
    
    # Read labels from .txt files
    for name in glob.glob(label_path):
        with open(name) as f:
            # Create annot dict
            annot_instance = {}
            # Filename is the path to img
            annot_instance['filename'] = images_path+name[:-3].split('/')[-1]+'jpg'
            image = cv2.imread(annot_instance['filename'])
            h, w = image.shape[:2]
            # Height and width of the img
            annot_instance['height'] = h
            annot_instance['width'] = w
            # ann dict inside first dict with boxes and labels for use and boxes and labels for ignore
            annot_instance['ann'] = {}
            f = [x.rstrip('\n').split() for x in f.readlines()]
            kk = [[float(y) for y in x[1:]] for x in f]
            ll = [label_dict[x[0]] for x in f]
            if len(ll)>0:
                annot_instance['ann']['bboxes'] = (np.array(kk)).astype(np.float32)
                annot_instance['ann']['labels'] = (np.array(ll)).astype(np.int64)
                # If there are no boxes to ignore, this is need to be a zero coordinate vector and empty label vector
                annot_instance['ann']['bboxes_ignore'] = np.zeros((0, 4), dtype=np.float32)
                annot_instance['ann']['labels_ignore'] = (np.array([])).astype(np.int64)
                if train_data:
                    annotations_train.append(annot_instance)
                else:
                    annotations_val.append(annot_instance)
    if train_data:
        mmcv.dump(annotations_train, save_train_path)
        print('Annot train ready: {}, len {}'.format(save_train_path, len(annotations_train)))
    else:
        mmcv.dump(annotations_val, save_val_path)
        print('Annot val ready: {}, len {}'.format(save_val_path, len(annotations_val)))      
        
def look_at_imgs_shapes(images_path):
    """Collect sizes of images.
    """
    label_path = images_path+'Label/*.txt'
    hh = []
    ww = []
    for name in glob.glob(label_path):
        with open(name) as f:
            annot_instance = {}
            annot_instance['filename'] = images_path+name[:-3].split('/')[-1]+'jpg'
            image = cv2.imread(annot_instance['filename'])
            h, w = image.shape[:2]
            annot_instance['height'] = h
            annot_instance['width'] = w
            hh.append(h)
            ww.append(w)
    return hh, ww