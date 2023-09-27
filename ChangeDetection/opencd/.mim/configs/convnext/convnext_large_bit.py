_base_ = [
    '../_base_/datasets/levir_cd.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
bit_norm_cfg = dict(type='LN', requires_grad=True)
pretrained = '/share/home/dongzhe/Dongzhe/Foundation_Model/Downstream_Tasks/semantic_segmentation/pretrained/epoch10_convnext_large.pth'  # noqa
model = dict(
    type='SiamEncoderDecoder',
    pretrained=pretrained,
    backbone=dict(
        type='ConvNeXt',
        in_chans=3,
        depths=[3, 3, 27, 3],
        dims=[192, 384, 768, 1536],
        drop_path_rate=0.4,
        layer_scale_init_value=1.0,
        out_indices=[0, 1, 2, 3], ),
    neck=dict(
        type='FeatureFusionNeck',
        policy='concat',
        out_indices=(0,)),
    decode_head=dict(
        type='BITHead',
        in_channels=192,
        channels=32,
        embed_dims=64,
        enc_depth=1,
        enc_with_pos=True,
        dec_depth=8,
        num_heads=8,
        drop_rate=0.,
        use_tokenizer=True,
        token_len=4,
        upsample_size=4,
        num_classes=2,
        norm_cfg=bit_norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

crop_size = (256, 256)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='MultiImgLoadImageFromFile'),
    dict(type='MultiImgLoadAnnotations'),
    dict(type='MultiImgRandomRotate', prob=0.5, degree=180),
    dict(type='MultiImgRandomCrop', crop_size=crop_size),
    dict(type='MultiImgRandomFlip', prob=0.5, direction='horizontal'),
    dict(type='MultiImgRandomFlip', prob=0.5, direction='vertical'),
    dict(type='MultiImgExchangeTime', prob=0.5),
    dict(
        type='MultiImgPhotoMetricDistortion',
        brightness_delta=10,
        contrast_range=(0.8, 1.2),
        saturation_range=(0.8, 1.2),
        hue_delta=10),
    dict(type='MultiImgNormalize', **img_norm_cfg),
    dict(type='MultiImgDefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='MultiImgLoadImageFromFile'),
    dict(
        type='MultiImgMultiScaleFlipAug',
        img_scale=(256, 256),
        # img_ratios=[0.75, 1.0, 1.25],
        flip=True,
        transforms=[
            dict(type='MultiImgResize', keep_ratio=True),
            dict(type='MultiImgRandomFlip'),
            dict(type='MultiImgNormalize', **img_norm_cfg),
            dict(type='MultiImgImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        img_dir='crop_256/trainval',
        ann_dir='crop_256/trainval/label',
        pipeline=train_pipeline),
    val=dict(
        img_dir='crop_256/test',
        ann_dir='crop_256/test/label',
        pipeline=test_pipeline),
    test=dict(
        img_dir='crop_256/test',
        ann_dir='crop_256/test/label',
        pipeline=test_pipeline))

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
    ])

# optimizer
optimizer = dict(
    constructor='LearningRateDecayOptimizerConstructor',
    _delete_=True,
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg={
        'decay_rate': 0.9,
        'decay_type': 'stage_wise',
        'num_layers': 12
    })

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

runner = dict(type='IterBasedRunner', max_iters=20000)
checkpoint_config = dict(by_epoch=False, interval=4000)
evaluation = dict(interval=4000, metric=['mFscore', 'mIoU'], pre_eval=True, save_best='Fscore.changed', greater_keys=['Fscore'])