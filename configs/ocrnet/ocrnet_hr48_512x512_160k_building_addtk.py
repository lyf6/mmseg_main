_base_ = './ocrnet_hr48_512x1024_160k_cityscapes.py'
#dataset
dataset_type = 'Mydata'
data_root = 'data/buildings/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(768, 768), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        #img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

whu_part1=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='whu/part1/created/img',
        ann_dir='whu/part1/created/mask',
        pipeline=train_pipeline
)
whu_part2=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='whu/part2/created/img',
        ann_dir='whu/part2/created/mask',
        pipeline=train_pipeline
)
whu_part3=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='whu/part3/created/img',
        ann_dir='whu/part3/created/mask',
        pipeline=train_pipeline
)
whu_part4=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='whu/part4/created/img',
        ann_dir='whu/part4/created/mask',
        pipeline=train_pipeline
)
inria=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='inria/created/img',
        ann_dir='inria/created/mask',
        pipeline=train_pipeline
)
mapchallenge1=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='mapchallenge/created/img',
        ann_dir='mapchallenge/created/mask',
        pipeline=train_pipeline,
        seg_map_suffix='.png',
        img_suffix='.jpg'
)
mapchallenge2=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='mapchallenge/valcreated/img',
        ann_dir='mapchallenge/valcreated/mask',
        pipeline=train_pipeline,
        seg_map_suffix='.png',
        img_suffix='.jpg'
)
mass=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='mass/created/img',
        ann_dir='mass/created/mask',
        pipeline=train_pipeline
)
open_cities_cities=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='Open_Cities_AI/created/img',
        ann_dir='Open_Cities_AI/created/mask',
        pipeline=train_pipeline
)
spacenet=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='spacenet/created/img',
        ann_dir='spacenet/created/mask',
        pipeline=train_pipeline
)
spacenet2=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='spacenet2/created/img',
        ann_dir='spacenet2/created/mask',
        pipeline=train_pipeline
)
spacenet4=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='spacenet4/created/img',
        ann_dir='spacenet4/created/mask',
        pipeline=train_pipeline
)

tk=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='tk/created/img',
        ann_dir='tk/created/mask',
        pipeline=train_pipeline
)

extra_add=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='extra_add/created/img',
        ann_dir='extra_add/created/mask',
        pipeline=train_pipeline
)


extra_damo=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='extra_damo/created/img',
        ann_dir='extra_damo/created/mask',
        pipeline=train_pipeline
)


luoliang=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='luoliang/created/img',
        ann_dir='luoliang/created/mask',
        pipeline=train_pipeline
)

hyj=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='hyj/created/img',
        ann_dir='hyj/created/mask',
        pipeline=train_pipeline
)

changping=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='changping/created/img',
        ann_dir='changping/created/mask',
        pipeline=train_pipeline
)

tjin=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='tjin/created/img',
        ann_dir='tjin/created/mask',
        pipeline=train_pipeline
)


val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='whu/part4/bcdd/two_period_data/div_images',
        ann_dir='whu/part4/bcdd/two_period_data/div_labels',
        pipeline=test_pipeline
)

test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='test',
        pipeline=test_pipeline   
)

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=1,
    train=[ tk,extra_add,extra_damo, luoliang, hyj, changping, tjin],
    val=val,
    test=test
    )



# optimizer
optimizer = dict(type='SGD', lr=0.0005, momentum=0.9, weight_decay=0.0005)  # set learning rate
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=5e-5, by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=20000) # set the number of the training iterations
# runtime settings

checkpoint_config = dict(by_epoch=False, interval=20000)
evaluation = dict(interval=5000, metric='mIoU')
norm_cfg = dict(type='SyncBN', requires_grad=True)

model = dict(
    pretrained= None,
    backbone=dict(
        extra=dict(
            stage2=dict(num_channels=(48, 96)),
            stage3=dict(num_channels=(48, 96, 192)),
            stage4=dict(num_channels=(48, 96, 192, 384)))),
    decode_head=[
        dict(
            type='FCNHead',
            in_channels=[48, 96, 192, 384],
            channels=sum([48, 96, 192, 384]),
            input_transform='resize_concat',
            in_index=(0, 1, 2, 3),
            kernel_size=1,
            num_convs=1,
            norm_cfg=norm_cfg,
            concat_input=False,
            dropout_ratio=-1,
            num_classes=2,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
            type='OCRHead',
            in_channels=[48, 96, 192, 384],
            channels=512,
            ocr_channels=256,
            input_transform='resize_concat',
            in_index=(0, 1, 2, 3),
            norm_cfg=norm_cfg,
            dropout_ratio=-1,
            num_classes=2,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))
    ])
