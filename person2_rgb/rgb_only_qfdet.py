import torch
from mmdet.models.builder import DETECTORS
from mmdet.models.detectors.qfdet import QFDet

@DETECTORS.register_module()
class RGBOnlyQFDet(QFDet):
    """Subclass of QFDet that skips the thermal backbone forward pass and zero-masks thermal features,
    complying with modularity constraints and providing a true RGB-only forward pass.
    """
    def __init__(self, *args, **kwargs):
        super(RGBOnlyQFDet, self).__init__(*args, **kwargs)
        # Disable the thermal backbone gradients and freeze it
        for param in self.backbone_t.parameters():
            param.requires_grad = False

    def extract_feat(self, img):
        """Directly extract features from the visible backbone+neck, and return zero features for thermal."""
        if isinstance(img, torch.Tensor):
            v_img, t_img = img, img
        else:
            v_img, t_img = img
        
        # Forward pass on visible branch
        v_feats = self.backbone(v_img)
        if self.with_neck:
            v_feats = self.neck(v_feats)
            
        # Create zero-tensors matching the visible feature maps shape/device/type
        t_feats = []
        for v_feat in v_feats:
            t_feats.append(torch.zeros_like(v_feat))
            
        return (v_feats, tuple(t_feats))
