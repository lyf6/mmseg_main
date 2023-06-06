# coding: utf-8
import cv2
import os
import numpy as np
import csv
from PIL import Image
import shapefile
from tqdm import tqdm
from osgeo import gdal
from building_footprint import get_polygons
import argparse
import shutil

def parse_args():
    parser = argparse.ArgumentParser(description='seg 2 shp')
    parser.add_argument('--segdir', help='the path to the seg result tif')
    parser.add_argument('--target_dir', help='the dir to save shps')
    args = parser.parse_args()
    return args


def draw_approx_hull_polygon(img, cnts):
    # img = np.copy(img)
    w, h = img.shape
    img = np.zeros(shape=(w,h, 3), dtype=np.uint8)
    cv2.drawContours(img, cnts, -1, (255, 0, 0), 2)  # blue
    return img

def segs2shps():
    args = parse_args()
    segdir=args.segdir
    target_dir=args.target_dir
    # clsmap = args.clsmap
    seg_ls = os.listdir(segdir)
    # cls_id = {}
    # content = open(clsmap).readlines()
    # for line in content:
    #     id, name = int(line.split(':')[0]), line.split(':')[1]
    #     cls_id[id] = name

    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir)
    for seg in tqdm(seg_ls):
        if seg.endswith(('.tif', '.tiff')):
            seg_path = os.path.join(segdir, seg)
            tareget_shape = os.path.join(target_dir, seg.split('.tif')[0]+'.shp')
            get_polygons(pred_masks_path=seg_path, polygons_path=tareget_shape)
            #seg2shp(cls_id, seg_path, tareget_shape)

def seg2shp(cls_id, imgpath, tareget_shape):
    w = shapefile.Writer(tareget_shape)
    w.autoBalance = 1
    w.field('class_name', 'C', '40')
    #print(img_name)
    tif = gdal.Open(imgpath)
    gt = tif.GetGeoTransform()
    prj = tif.GetProjection()
    img = np.array(tif.ReadAsArray())
    f = open(tareget_shape+".prj", 'w')
    f.write(prj)
    f.close()
    print('processing '+ imgpath)
    x_min = gt[0]
    x_size = gt[1]
    y_min = gt[3]
    y_size = gt[5]
    for key, value in cls_id.items():
        index = img==key
        tmp_img = np.zeros_like(img)
        tmp_img[index] = 255
        ret, thresh = cv2.threshold(tmp_img, 0, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        refine_cnt = []
        for cnt in contours:
            area =cv2.contourArea(cnt)
            #print(cnt)
            if area >30:
                refine_cnt.append(cnt)
        #tmp_str = ''
        if len(refine_cnt)>0:
            for cnt in tqdm(refine_cnt):
                #print(cnt)
                #tmp_str = img_name + ' ' + value
                #print(tmp_str)
                point_list = []
                polygon_list = []
                point = []
                for loc in cnt:
                    x = loc[0][0]*x_size + x_min
                    y = loc[0][1]*y_size + y_min
                    point.append(float(x))
                    point.append(float(y))
                    point_list.append(point)
                    point = []
                polygon_list.append(point_list)
                w.poly(polygon_list)
                w.record(value)
    w.close()          

# segdir = 'C://code//building_extraction//output//combine_crop_seg'
# target_dir = 'C://code//building_extraction//output//shp'
# segs2shps(segdir, target_dir)
if __name__ == '__main__':
    segs2shps()