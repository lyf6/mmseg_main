#whole test, including the following steps:
#  step 1: divided the whole image into small crops
#  setp 2: semantic segmentation on the crops extracting the crop classification map
#  step 3: combine the crop classification map to the whole image classification map
#  step 4: convert the image classification map to shp
#!/usr/bin/env bash

GPU_NUM=10 # set the number of gpus to use
CONFIG_FILE=configs/segformer/segformer_mit-b5_8xb1-160k_corn-1024x1024.py
#configs/ocrnet/ocrnet_hr48_512x512_160k_greenhouse.py
#configs/ocrnet/ocrnet_hr48_512x512_160k_cd.py
# configs/ocrnet/ocrnet_hr48_512x512_160k_greenhouse.py
#configs/ocrnet/ocrnet_hr48_512x512_160k_building.py
# configs/ocrnet/ocrnet_hr48_512x512_160k_cd.py
#configs/ocrnet/ocrnet_hr48_512x512_160k_road.py
#configs/ocrnet/ocrnet_hr48_512x512_160k_building.py
#configs/ocrnet/ocrnet_hr48_512x512_160k_cd.py
#configs/ocrnet/ocrnet_hr48_512x512_160k_car.py
#configs/ocrnet/ocrnet_hr48_512x512_160k_cd.py
#ocrnet_hr48_512x512_160k_building_add_uav.py
#ocrnet_hr48_512x512_10k_factory_building.py
#ocrnet_hr48_512x512_160k_building_addtk.py
#/home/yf/Documents/mmsegmentation/configs/ocrnet/ocrnet_hr48_512x512_160k_road.py
#/home/yf/Documents/mmsegmentation/configs/ocrnet/ocrnet_hr48_512x512_160k_building_addtk.py
work_dir=work_dirs/segformer_mit-b5_8xb1-160k_corn-1024x1024
#/buildings_09_27_ocrnet
# dachangzhen_road_ocrnet
# buildings_09_09_ocrnet
#work_dirs/dachangzhen_road_ocrnet
#dachangzhen_building_ocrnet
#greenhouse_ocrnet
#cd_ocrnet
#building_07_21_luoliang_ocrnet
# work_dirs/buildings_09_27_ocrnet
CHECKPOINT_FILE=${work_dir}/iter_40000.pth
testdir=/home/yf/disk/feilianghua/corn_test
#/home/yf/disk/changedetection/big_tif
#/home/yf/disk/tmp/car
#/home/yf/disk/changedetection/tif_imgs #set the path to test
target_dir=~/disk/res
combine_dir=${target_dir}/corn
shp_dir=${target_dir}/shp
tmpdir=${target_dir}
crop_img=crop_imgs
crop_seg=${target_dir}/seg1
crop_size=1024
clsname=/home/yf/Documents/mmseg/clsname.txt
img_suffix=.tif
clsmap=./clsmap.txt
#step 1 divided the images into crops, and save it in target_dir
python myutiles/divided.py --imgpath ${testdir} --target_dir ${tmpdir}/${crop_img} --crop_size ${crop_size} --stride ${crop_size}

# # step 2 semantic segmentation on the crops, and save the images in crop_res directory

# ./tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} --cfg-options "data.test.data_root=${tmpdir} data.test.img_dir=${crop_img}  data.test.classes=${clsname} data.test.img_suffix=${img_suffix}" --show-dir ${crop_seg}
python tools/test.py ${CONFIG_FILE} \
    ${CHECKPOINT_FILE} \
    --cfg-options val_dataloader.dataset.data_root=${tmpdir} val_dataloader.dataset.data_prefix.img_path=${crop_img}  val_dataloader.dataset.metainfo.classes="[bg, corn]" val_dataloader.dataset.img_suffix=${img_suffix} \
    --show-dir ${crop_seg}
# step 3 combine the crop map to whole map
python ./myutiles/combine.py --reference_imgs ${testdir} --segpngs ${crop_seg} --crop_size ${crop_size} --save_dir ${combine_dir} --stride ${crop_size}
# python ./myutiles/rewrite.py --reference_imgs ${testdir} --segpngs ${crop_seg}  --save_dir ${combine_dir}
# step4 convert the mask 2 shp 
# python ./myutiles/seg2shape.py --segdir ${combine_dir} --target_dir ${shp_dir} --clsmap ${clsmap}
# python ./myutiles/seg2shape_building.py --segdir ${combine_dir} --target_dir ${shp_dir}
