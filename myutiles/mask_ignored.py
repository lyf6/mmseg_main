from __future__ import division
import numpy as np
import os
from PIL import Image
from math import *
import gdal, ogr, os, osr
from tifffile import imread
Image.MAX_IMAGE_PIXELS = None
import argparse
from myutiles.divided import array2raster

def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('--reference_masks', help='the path to the reference images')
    parser.add_argument('--pred_masks', help='the dir to save crop segmentation images')
    parser.add_argument('--ignored_index', type=int, help='the number of class')
  
    args = parser.parse_args()
    return args


def get_confusion_matrix(label, pred_label, num_class, ignore_index=5):
    """
    Calcute the confusion matrix by given label and pred
    """


    label = label.reshape(-1)
    index = label!=255
    label = label[index]
    pred_label = pred_label.reshape(-1)
    pred_label = pred_label[index]
    index = (label * num_class + pred_label).astype('int32')

    label_count = np.bincount(index)
    confusion_matrix = np.zeros((num_class, num_class))

    for i_label in range(num_class):
        for i_pred in range(num_class):
            cur_index = i_label * num_class + i_pred
            if cur_index < len(label_count):
                confusion_matrix[i_label,
                                 i_pred] = label_count[cur_index]
    return confusion_matrix




def mask_ignored():
    
    args = parse_args()
    dir = args.reference_masks
    pred_masks = args.pred_masks
    files=os.listdir(dir)
    

    for file in files:
        #print(file)
        gt_mask_path = os.path.join(dir, file)
        pred_mask_path = os.path.join(pred_masks, file.split('.ti')[0] +  'pred.tif')
        #print(pred_mask)
        tif = gdal.Open(pred_mask_path)
        prj = tif.GetProjection()
        dtype = "Byte"
        gt = tif.GetGeoTransform()
        gt_mask = np.array(gdal.Open(gt_mask_path).ReadAsArray())
        pred_mask = np.array(tif.ReadAsArray())
        #print(gt_mask.shape)
        index = gt_mask==args.ignored_index
        pred_mask[index] = 255
        gt_mask[index] = 255
        array2raster(pred_mask_path, pred_mask, dtype, gt, prj)
        array2raster(gt_mask_path, gt_mask, dtype, gt, prj)






if __name__ == '__main__':
    
    mask_ignored()