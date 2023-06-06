import os
from PIL import Image
from math import *
Image.MAX_IMAGE_PIXELS = None
import argparse
import shutil
from osgeo import gdal, osr
import numpy as np
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('--imgpath', help='the path to the divided image')
    parser.add_argument('--target_dir', help='the dir to save crop images')
    parser.add_argument(
        '--crop_size', type=int, default=512, help='the checkpoint file to load weights from')
    parser.add_argument(
        '--stride', type=int, default=256, help='the checkpoint file to load weights from')
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
    GDT_dtype = gdal.GDT_Unknown
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


def divided_tif():

    args = parse_args()
    imgpath_dir=args.imgpath
    target_dir=args.target_dir
    crop_size = args.crop_size
    stride = args.stride
    
    
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir)
    weight = crop_size
    height = crop_size
   
    if os.path.isfile(imgpath_dir):
        files = [imgpath_dir]
    else:
        files = [os.path.join(imgpath_dir, file) for file in os.listdir(imgpath_dir)]
     
    for imgpath in files:
        #imgpath = os.path.join(imgpath_dir, path)
        tif = gdal.Open(imgpath)
        prj = tif.GetProjection()
        dtype = "Byte"
        gt = tif.GetGeoTransform()
        x_min = gt[0]
        pixelWidth = gt[1]
        y_min = gt[3]
        pixelHeight = gt[5]
        b = gt[2]
        d = gt[4]
        img = np.array(tif.ReadAsArray())
        if img.ndim > 2:
            img = img.transpose((1,2,0))
        if img.ndim == 2:
            band_num = 1
            ori_w, ori_h= img.shape
        else:
            band_num = img.shape[2]
            ori_w, ori_h, _ = img.shape

        pad_w = int(ceil(abs(ori_w-weight)/stride)*stride-ori_w + weight)
        pad_h = int(ceil(abs(ori_h-height)/stride)*stride-ori_h + height)
        print(img.shape)
        if  band_num > 1:
            pad_img=np.pad(img,((0,pad_w),(0,pad_h),(0,0)),'constant', constant_values=0)
            new_w, new_h,_= pad_img.shape
            # for rgb image, the non data is set as 0
            index = np.isnan(img)
            img[index] = 0
        else:
            pad_img=np.pad(img,((0,pad_w),(0,pad_h)),'constant', constant_values=255)
            new_w, new_h = pad_img.shape
            # for  image, the non data is set as 0
            index = np.isnan(img)
            img[index] = 255

        num_w = int((new_w-weight)/stride)+1
        num_h = int((new_h-height)/stride)+1
        for w_id in range(num_w):
            for h_id in range(num_h):
                if band_num > 1:
                    array=pad_img[w_id*stride:w_id*stride+weight, h_id*stride:h_id*stride+height,:]
                else:
                    array=pad_img[w_id*stride:w_id*stride+weight, h_id*stride:h_id*stride+height]
                rasterOrigin = (x_min+w_id*stride*pixelWidth,y_min+h_id*stride*pixelHeight)
                originX = x_min+h_id*stride*pixelWidth
                originY = y_min+w_id*stride*pixelHeight
                geotransform = [originX, pixelWidth, b, originY, d, pixelHeight]
                #originX, pixelWidth, b, originY, d, pixelHeight = geotransform
                newRasterfn = os.path.join(target_dir, os.path.basename(imgpath).split('.')[0]+'_'+str(w_id)+'_'+str(h_id)+'.tif')
                print(newRasterfn)
                array2raster(newRasterfn, array, dtype, geotransform, prj)





def divided():
    
    args = parse_args()
    dir=args.imgpath
    target_dir=args.target_dir
    crop_size = args.crop_size
    if os.path.isfile(dir):
        files = [dir]
    else:
        files = [os.path.join(dir, file) for file in os.listdir(dir)]
    #print('files')
    weight=crop_size
    height=crop_size
    stride=crop_size

    if not os.path.isdir(target_dir):
        os.makedirs(target_dir)

    for file in files:
        path=file
        #print(file)
        basename = os.path.basename(file)
        img=np.array(Image.open(path))
        index = np.isnan(img)
        img[index] = 0
        values = np.unique(img)
        #print(values)
        #img[img==5] = 255
        #values = np.unique(img)
        #print(values)
        ori_w, ori_h= img.shape[0],img.shape[1]
        pad_w = int((ceil(ori_w/stride))*stride-ori_w)
        pad_h = int((ceil(ori_h/stride))*stride-ori_h)
        #print(pad_w)
        pad_img=np.pad(img,((0,pad_w),(0,pad_h),(0,0)),'constant', constant_values=0)
        new_w, new_h = pad_img.shape[0], pad_img.shape[1]
        num_w = int(new_w/weight)
        num_h = int(new_h/height)
        #count=0
        #print(num_w)
        for w_id in range(num_w):
            for h_id in range(num_h):
                tmp_img=pad_img[w_id*stride:w_id*stride+weight, h_id*stride:h_id*stride+height]
                tmp_img=Image.fromarray(tmp_img)
                target_path=basename.split('.')[0]+'_'+str(w_id)+'_'+str(h_id)+'.tif'
                target_path=os.path.join(target_dir,target_path)
                tmp_img.save(target_path)
if __name__ == '__main__':
   
    divided_tif()