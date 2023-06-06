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
   
        tmp_path = file.split('.tif')[0]   + '.png'
        print(tmp_path)
        tmp_path = os.path.join(result_path,tmp_path)
        
        tmp_img = np.array(Image.open(tmp_path), dtype=np.uint8)
        target_path = file.split('.ti')[0] +  'pred.tif'
        target_path = os.path.join(target_dir,target_path)
        array2raster(target_path, tmp_img, dtype, gt, prj)

if __name__ == '__main__':
    combine()