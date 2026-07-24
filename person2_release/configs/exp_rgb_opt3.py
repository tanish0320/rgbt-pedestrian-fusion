_base_ = './exp_rgb_only.py'
model = dict(
    test_cfg=dict(
        nms=dict(type='nms', iou_threshold=0.60)
    )
)
