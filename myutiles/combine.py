from __future__ import division
import numpy as np
import os
from PIL import Image
from math import *
from osgeo import gdal, osr
from tifffile import imread
Image.MAX_IMAGE_PIXELS = None
import argparse
import shutil

def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('--reference_imgs', help='the path to the reference images')
    parser.add_argument('--segpngs', help='the dir to save crop segmentation images')
    parser.add_argument(
        '--crop_size', default=512, type=int, help='the checkpoint file to load weights from')
    parser.add_argument(
        '--stride', default=256, type=int, help='the stride')
    parser.add_argument(
        '--num_cls', default=2, type=int, help='the number of class')    
    parser.add_argument(
        '--save_dir', help='the directory to save the seg results')
    args = parser.parse_args()
    return args


def array2raster(newRasterfn, array, dtype, geotransform, prj):
    """
    save GTiff file from numpy.array
    input:
        newRasterfn: save file name
        dataset : original tif file
        array : numpy.array
        dtype: Byte or Float32.
    """
    cols = array.shape[1]
    rows = array.shape[0]
    originX, pixelWidth, b, originY, d, pixelHeight = geotransform

    driver = gdal.GetDriverByName('GTiff')

    # set data type to save.
    # GDT_dtype = gdal.GDT_Unknownreset

    if dtype == "Byte":
        GDT_dtype = gdal.GDT_Byte
    elif dtype == "Float32":
        GDT_dtype = gdal.GDT_Float32
    else:
        print("Not supported data type.")

    # set number of band.
    if array.ndim == 2:
        band_num = 1
    else:
        band_num = array.shape[2]

    outRaster = driver.Create(newRasterfn, cols, rows, band_num, GDT_dtype)
    outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))

    # Loop over all bands.
    for b in range(band_num):
        outband = outRaster.GetRasterBand(b + 1)
        # Read in the band's data into the third dimension of our array
        if band_num == 1:
            outband.WriteArray(array)
        else:
            outband.WriteArray(array[:,:,b])

    outRasterSRS = osr.SpatialReference(wkt=prj)
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outband.FlushCache()


# dir='/home/yf/disk/myannotated_cd/rgbp'
# target_dir='/home/yf/disk/myannotated_cd/prod_combv2'
# result_path= '/home/yf/disk/myannotated_cd/rgbp-res'
def combine():
    args = parse_args()
    dir = args.reference_imgs

    #'/home/yf/disk/gid/test/images'
    #'/home/yf/disk/whu/part4/bcdd/two_period_data/test'
    #'/home/yf/disk/myannotated_cd/2017-2019val/imgs'
    # '/home/yf/disk/whu/part4/bcdd/two_period_data/before/image'
    target_dir = args.save_dir 
    #'/home/yf/Documents/mmsegmentation/work_dirs/gid_fcn/pred'
    #'/home/yf/disk/myannotated_cd/prod_comb'
    #'/home/yf/disk/whu/part4/bcdd/two_period_data/combine'
    result_path = args.segpngs

    # if not os.path.isdir(result_path):
    #     os.makedirs(target_dir)
    #'/home/yf/Documents/mmsegmentation/work_dirs/gid_fcn/results'
    #'/home/yf/disk/myannotated_cd/test_masks'
        #'/home/yf/disk/output/myann/seg_hrnet_w48_trainval_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484x2/test_results'
    files=os.listdir(dir)
    crop_size = args.crop_size
    weight=crop_size
    height=crop_size
    stride=args.stride
    num_cls = args.num_cls

    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir)
    


    for file in files:
        #if file=='bdf.tif':
        #path=os.path.join(dir,file)
        #print(path)
        imgpath = os.path.join(dir, file)
        print(imgpath)
        tif = gdal.Open(imgpath)
        prj = tif.GetProjection()
        dtype = "Byte"
        gt = tif.GetGeoTransform()
        # x_min = gt[0]
        # pixelWidth = gt[1]
        # y_min = gt[3]
        # pixelHeight = gt[5]
        # b = gt[2]
        # d = gt[4]
        img = np.array(tif.ReadAsArray())
        img = img.transpose((1,2,0))
        # img = np.array(Image.open(path))
        ori_w, ori_h,_ = img.shape

        # pad_w = int(ceil(ori_w/stride)*stride)
        # pad_h = int(ceil(ori_h/stride)*stride)
        # print(pad_w, pad_h)
        pad_w = int(ceil(abs(ori_w-weight)/stride)*stride-ori_w + weight)
        pad_h = int(ceil(abs(ori_h-height)/stride)*stride-ori_h + height)
        #print(ceil(ori_w/stride), ceil(ori_h/stride))
        #print(ori_w/stride, ori_h/stride)
        logit = np.zeros(shape=(ori_w+pad_w, ori_h+pad_h), dtype=np.uint8)
        # pad_img=np.pad(img,((0,pad_w),(0,pad_h)),'constant', constant_values=255) # all ignored labeld default 255
        new_w, new_h= logit.shape
        num_w = int((new_w-weight)/stride)+1
        num_h = int((new_h-height)/stride)+1
        print(num_w,num_h)
        for w_id in range(num_w):
            for h_id in range(num_h):
                tmp_path = file.split('.tif')[0]  + '_'+str(w_id)+'_'+str(h_id) + '.tif'
                print(tmp_path)
                tmp_path = os.path.join(result_path,tmp_path)
                # tmp_img = np.load(tmp_path)
                # tmp_img = tmp_img.transpose((1,2,0))
                tmp_img = np.array(Image.open(tmp_path),dtype=np.uint8)
                logit[w_id * stride:w_id * stride + weight, h_id * stride:h_id * stride + height] = tmp_img
                # count=count+1
        # pad_img = np.argmax(logit, axis=2)
        pad_img=logit[:ori_w,:ori_h]
        target_path = file.split('.ti')[0] +  'pred.tif'
        target_path = os.path.join(target_dir,target_path)
        array2raster(target_path, pad_img, dtype, gt, prj)

if __name__ == '__main__':
    combine()