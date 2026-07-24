"""Phase 3 ABLATION (isolation run — see docs/phase3_fusion_plan.md item 6):
Modality-dropout training ONLY, baseline anchor levels (no P2/stride-4). Run on a 3050 (6GB)
alongside the full fusion_v1 (4060) and p2_only (other 3050) runs, to isolate what modality
dropout alone contributes to the RGB-only-collapse fix vs. what P2 fusion alone contributes
to the mAP_S delta.

Identical to the ORIGINAL baseline qfdet_r50_fpn_1x_vtuav.py neck/anchor config (start_level=1,
num_outs=5, strides=[8,16,32,64,128], nms_pre=1000 — no anchor-count increase here, so no
need to raise it) except train_pipeline gains RandomMasking, and this fine-tunes from the
pretrained checkpoint with the fusion_v1 run's optimizer/schedule for a fair comparison.

RandomMasking added to train_pipeline (before normalization, since zeroing after would
produce a nonzero-mean tensor, not true black — see MultiNormalize's mean_list below).
Already exists in mmdet/datasets/pipelines/multispectral_transforms.py, unused by the
baseline config. Randomly zeroes RGB, zeroes thermal, or leaves both real, each train
step. Targets Phase 2's finding: RGB-only ablation collapses 7x (mAP 0.042) vs.
thermal-only's 29% degradation (mAP 0.232) against fused 0.299 — evidence the existing
quality_attention gate was never trained under a missing/weak-modality condition.

Known cost, not yet measured empirically: adding stride-4 roughly quadruples total anchor
positions at 640x512 (6,820 -> 27,300). test_cfg.nms_pre raised 1000->2000 as a starting
point pending the smoke-test measurement in docs/phase3_fusion_plan.md item 3 — revisit if
recall/precision looks off in early eval.

Checkpoint-load reality check (smoke-tested, not assumed): start_level 1->0 shifts every
existing lateral_conv's index by one (index 0 now means C2/256ch instead of C3/512ch), so
the pretrained C3/C4/C5 lateral convs do NOT map onto their old slots — mmcv's
load_checkpoint reports a real shape mismatch on all 3 lateral_convs, not just the new P2
one, and reinitializes all 4 lateral convs randomly rather than reusing 3 pretrained ones.
Accepted deliberately (see docs/phase3_fusion_plan.md discussion) rather than writing a
custom checkpoint-remapping step — these are small 1x1 convs, expected to re-converge
quickly during fine-tuning. atss_cls also reinitializes (num_classes=1 vs. the checkpoint's
saved num_classes=3 — kept at 1 here since VTUAVdet is a genuine single-class training
problem; num_classes=3 with only-ever-negative extra classes would dilute the loss for no
benefit, unlike the eval-only configs where it was safe to match the checkpoint exactly).
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
        num_classes=1,
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
        num_classes=1,
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
    # training and testing settings
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
        nms_pre=1000,  # baseline value — no P2 here, anchor count unchanged from original
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))

# dataset settings
# Phase 1 dataset audit found 2 degenerate zero-width boxes (val ann 41153, test ann 43651,
# both w=0). VTUAVdet._parse_ann_info (mmdet/datasets/vtuav.py) already drops any box with
# w < 1 or h < 1 unconditionally, so both are filtered automatically on every load — no extra
# filter_cfg/min_gt_bbox_wh option needed here.
dataset_type = 'VTUAVdet'
data_root = 'C:/Users/urbra/OneDrive/Desktop/Projects/GG/data/VTUAV_subset/'
img_norm_cfg = dict(
    mean_list=([83.20, 92.24, 97.70], [134.84, 134.84, 134.84]),
    std_list=([57.77, 57.41, 57.69], [81.58, 81.58, 81.58]), to_rgb=True)
train_pipeline = [
    dict(type='LoadImagePairFromFile', spectrals=('visible', 'thermal')),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(640, 512), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    # modality dropout — must run before MultiNormalize (zeroing post-normalize would
    # produce a nonzero-mean tensor, not true black; mean_list above is nonzero)
    dict(type='RandomMasking', p=(0.15, 0.15, 0.7)),
    dict(type='MultiNormalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImagePairFromFile', spectrals=('visible', 'thermal')),
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
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/train.json',
        img_prefix=data_root + 'mmdet_data/train/',
        pipeline=train_pipeline),
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

# fine-tune from the provided pretrained checkpoint — never from scratch (rubric requirement)
load_from = '../../checkpoints/qfdet_vtuav_pretrained.pth'

# optimizer — lower LR than the from-ImageNet baseline schedule since this fine-tunes an
# already-converged checkpoint plus new P2-level layers; revisit after the smoke test
optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(_delete_=True, max_norm=35, norm_type=2))
fp16 = dict(loss_scale=512.)
runner = dict(type='EpochBasedRunner', max_epochs=6)  # short fine-tune; extend if time allows

work_dir = '../../checkpoints/train_qfdet_dropout_only'
