"""Stage 2 eval-only config: Thermal-only ablation of the pretrained QFDet checkpoint.
RGB stream is zeroed via ZeroModality right after loading (same pretrained
dual-stream weights, same architecture, one input ablated) so this evaluates the
provided pretrained weights as the rubric specifies, not a retrained substitute.
No training. num_classes=3 matches the checkpoint's own embedded training config
exactly (mmdet-rgbtdroneperson/checkpoints/qfdet_vtuav.pth), so every layer loads
with zero shape mismatch. Class 0 ('person') is the one that matters for VTUAV_subset.
"""
_base_ = [
    '../configs/_base_/datasets/coco_detection.py',
    '../configs/_base_/schedules/schedule_1x.py', '../configs/_base_/default_runtime.py'
]
model = dict(
    type='QFDet',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5),
    bbox_head=dict(
        type='ATSSQHead',
        num_classes=3,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        centerness=1,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    bbox_prehead=dict(
        type='QFDetPreHead',
        num_classes=3,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        centerness=1,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=0.5),
        loss_bbox=dict(type='GIoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=0.5),
        loss_quality=dict(type='MSELoss', loss_weight=0.5)),
    base_fusion='cat',
    quality_attention=True,
    poolupsample=1,
    reweight=True,
    train_cfg=dict(
        assigner=dict(type='QLSAssigner',
                      topk=9,
                      alpha=0.8,
                      quality='x',
                      iou_calculator=dict(type='BboxDistanceMetric'),
                      iou_mode='siwd',
                      overlap_mode='hybrid',
                      ),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))

dataset_type = 'VTUAVdet'
data_root = 'C:/Users/urbra/OneDrive/Desktop/Projects/GG/data/VTUAV_subset/'
img_norm_cfg = dict(
    mean_list=([83.20, 92.24, 97.70], [134.84, 134.84, 134.84]),
    std_list=([57.77, 57.41, 57.69], [81.58, 81.58, 81.58]), to_rgb=True)

test_pipeline = [
    dict(type='LoadImagePairFromFile', spectrals=('visible', 'thermal')),
    dict(type='ZeroModality', zero='rgb'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 512),
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

# NOTE: tools/test.py always builds its dataset from cfg.data.test (not
# cfg.data.val), regardless of --eval. This config evaluates the VAL split —
# both val and test dicts point at val.json so it's correct no matter which
# key a future script version reads. Use eval_qfdet_baseline_vtuav_test.py
# (data.test -> test.json) for the test-split numbers.
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=0,
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/val.json',
        img_prefix=data_root + 'mmdet_data/val/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/val.json',
        img_prefix=data_root + 'mmdet_data/val/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')

work_dir = '../../results/preds/eval_thermal_only_qfdet'
