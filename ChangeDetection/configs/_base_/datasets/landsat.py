# dataset settings
dataset_type = 'Landsat_Dataset'
data_root = 'data/Landsat'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (416, 416)
train_pipeline = [
    dict(type='MultiImgLoadImageFromFile'),
    dict(type='MultiImgMultiAnnLoadAnnotations', reduce_semantic_zero_label=True), # Zero label should be ignored when training semantic branches
    dict(type='MultiImgRandomCrop', crop_size=crop_size),
    dict(type='MultiImgRandomFlip', prob=0.5),
    dict(type='MultiImgNormalize', **img_norm_cfg),
    dict(type='MultiImgDefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg', 
        'gt_semantic_seg_from', 'gt_semantic_seg_to']),
]
test_pipeline = [
    dict(type='MultiImgLoadImageFromFile'),
    dict(
        type='MultiImgMultiScaleFlipAug',
        img_scale=(416, 416),
        # img_ratios=[0.75, 1.0, 1.25],
        flip=False,
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
        type=dataset_type,
        data_root=data_root,
        img_dir='train',
        ann_dir=dict(
            binary_dir='train/label',
            semantic_dir_from='train/label1',
            semantic_dir_to='train/label2'),
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='val',
        ann_dir=dict(
            binary_dir='val/label',
            semantic_dir_from='val/label1',
            semantic_dir_to='val/label2'),
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='val',
        ann_dir=dict(
            binary_dir='val/label',
            semantic_dir_from='val/label1',
            semantic_dir_to='val/label2'),
        pipeline=test_pipeline))