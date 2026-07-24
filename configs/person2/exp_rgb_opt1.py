_base_ = './exp_rgb_only.py'

img_norm_cfg = dict(
    mean_list=([83.20, 92.24, 97.70], [134.84, 134.84, 134.84]),
    std_list=([57.77, 57.41, 57.69], [81.58, 81.58, 81.58]), to_rgb=True)

# Validation pipeline with correct val directories
val_pipeline = [
    dict(type='LoadImagePairFromFile', spectrals=('VTUAV_co/val/images', 'VTUAV_ir/val/images')),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(960, 768),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='MultiNormalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img']),
        ])
]

# Test pipeline with correct test directories
test_pipeline = [
    dict(type='LoadImagePairFromFile', spectrals=('VTUAV_co/test/images', 'VTUAV_ir/test/images')),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(960, 768),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='MultiNormalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    val=dict(pipeline=val_pipeline),
    test=dict(pipeline=test_pipeline)
)
