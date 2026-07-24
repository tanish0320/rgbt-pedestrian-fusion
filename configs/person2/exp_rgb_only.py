_base_ = '../../qfdet_configs/qfdet_r50_fpn_1x_vtuav.py'

# Use the custom RGB-only subclass
model = dict(
    type='RGBOnlyQFDet'
)

# Register custom imports to load our subclass at runtime
custom_imports = dict(
    imports=['person2_rgb.rgb_only_qfdet'],
    allow_failed_imports=False
)
