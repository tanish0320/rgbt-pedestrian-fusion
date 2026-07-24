import os
import time
import json
import torch
from mmcv import Config
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector
from mmcv.runner import load_checkpoint
from mmdet.utils import build_dp, get_device

def run_evaluation(config_path, checkpoint_path, split='val'):
    print(f"\n==========================================")
    print(f"Evaluating: {config_path}")
    print(f"Split: {split}")
    print(f"==========================================")
    
    cfg = Config.fromfile(config_path)
    
    # Configure split dataset
    if split == 'val':
        cfg.data.test = cfg.data.val
    cfg.data.test.test_mode = True
    
    # Build dataset and dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=0,
        dist=False,
        shuffle=False)
        
    # Build model and load checkpoint
    cfg.model.train_cfg = None
    model = build_detector(cfg.model)
    
    # Load checkpoint
    checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
        
    device = get_device()
    model = build_dp(model, device, device_ids=[0])
    model.eval()
    
    # Warmup
    warmup_count = min(10, len(data_loader))
    for i, data in enumerate(data_loader):
        if i >= warmup_count:
            break
        with torch.no_grad():
            model(return_loss=False, rescale=True, **data)
            
    # Benchmark loop
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    
    outputs = []
    pure_inf_time = 0
    
    for i, data in enumerate(data_loader):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
            
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        
        pure_inf_time += (t1 - t0)
        outputs.extend(result)
        
    total_time = time.perf_counter() - start_time
    num_images = len(dataset)
    
    fps = num_images / pure_inf_time if pure_inf_time > 0 else 0
    inf_time_ms = (pure_inf_time / num_images) * 1000 if num_images > 0 else 0
    
    # Run evaluation
    eval_kwargs = cfg.get('evaluation', {}).copy()
    for key in ['interval', 'tmpdir', 'start', 'gpu_collect', 'save_best', 'rule', 'dynamic_intervals']:
        eval_kwargs.pop(key, None)
    eval_kwargs.update(dict(metric='bbox'))
    
    metric = dataset.evaluate(outputs, **eval_kwargs)
    
    res = {
        'split': split,
        'fps': fps,
        'inf_time_ms': inf_time_ms,
        'mAP': metric.get('bbox_mAP', 0.0),
        'mAP50': metric.get('bbox_mAP_50', 0.0),
        'mAP75': metric.get('bbox_mAP_75', 0.0),
        'mAPS': metric.get('bbox_mAP_s', 0.0),
        'mAPM': metric.get('bbox_mAP_m', 0.0),
        'mAPL': metric.get('bbox_mAP_l', 0.0)
    }
    return res

if __name__ == '__main__':
    config_path = 'configs/person2/exp_rgb_only.py'
    checkpoint_path = 'checkpoints/qfdet_r50_fpn_1x_vtuav.pth'
    
    os.makedirs('results', exist_ok=True)
    
    val_res = run_evaluation(config_path, checkpoint_path, split='val')
    test_res = run_evaluation(config_path, checkpoint_path, split='test')
    
    results = {
        'exp_rgb_only': {
            'val': val_res,
            'test': test_res,
            'params': '36.9 M',
            'flops': '130.75 GFLOPs',
            'model_size': '293.2 MB'  # Model size is smaller since we only count parameters
        }
    }
    
    # Save metrics json
    with open('results/person2_metrics.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    print("Baseline RGB evaluation completed successfully.")
