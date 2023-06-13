export CUDA_VISIBLE_DEVICES=1,2,3,4
CONFIG_FILE=configs/segformer/segformer_mit-b5_8xb1-160k_corn-1024x1024.py
GPU_NUM=4
tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM}