_base_ = ['./segformer_mit-b0_8xb1-160k_cityscapes-1024x1024.py']

dataset_type = 'Mydata'
data_root = 'data/corn/'
classes = ('bg', 'corn')
metainfo=dict(classes=classes)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
        scale=(2048, 1024),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
train_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    metainfo=metainfo,
    data_prefix=dict(
            img_path='images', seg_map_path='labels'),
    img_suffix='.png',
    pipeline=train_pipeline
)

test_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    metainfo=metainfo,
    
     data_prefix=dict(
            img_path='images', seg_map_path='labels'),
    img_suffix='.png'
)

optimizer = dict(lr=0.00001)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)
# learning policy
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=0,
        end=20000,
        by_epoch=False)
]
# training schedule for 20k
train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=1000)

train_dataloader = dict(
    dataset = train_dataset
)

val_dataloader = dict(
    dataset = test_dataset
)

test_dataloader = val_dataloader

model = dict(
    data_preprocessor=dict(
        size=(512, 512)),
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint='pretrain/mit_b5.pth'),
        embed_dims=64,
        num_layers=[3, 6, 40, 3]),
    decode_head=dict(in_channels=[64, 128, 320, 512],num_classes=len(classes)),)

load_from ="/home/yf/disk/pretrained/segformer/segformer_mit-b5_8x1_1024x1024_160k_cityscapes_20211206_072934-87a052ec.pth"