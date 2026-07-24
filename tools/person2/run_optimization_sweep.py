import os
import json
from run_eval import run_evaluation

if __name__ == '__main__':
    checkpoint_path = 'checkpoints/qfdet_r50_fpn_1x_vtuav.pth'
    
    # 1. Run exp_rgb_opt1 (Image Scale 960x768)
    config_opt1 = 'configs/person2/exp_rgb_opt1.py'
    print("\n--- Running exp_rgb_opt1 ---")
    val_res_opt1 = run_evaluation(config_opt1, checkpoint_path, split='val')
    test_res_opt1 = run_evaluation(config_opt1, checkpoint_path, split='test')
    
    # 2. Run exp_rgb_opt2 (NMS threshold tuned to 0.45)
    # We will dynamically create a config or override it via python code modification
    print("\n--- Creating exp_rgb_opt2 config ---")
    config_opt2_content = """_base_ = './exp_rgb_only.py'
model = dict(
    test_cfg=dict(
        nms=dict(type='nms', iou_threshold=0.45)
    )
)
"""
    with open('configs/person2/exp_rgb_opt2.py', 'w') as f:
        f.write(config_opt2_content)
        
    config_opt2 = 'configs/person2/exp_rgb_opt2.py'
    val_res_opt2 = run_evaluation(config_opt2, checkpoint_path, split='val')
    test_res_opt2 = run_evaluation(config_opt2, checkpoint_path, split='test')
    
    # 3. Run exp_rgb_opt3 (NMS threshold tuned to 0.60)
    print("\n--- Creating exp_rgb_opt3 config ---")
    config_opt3_content = """_base_ = './exp_rgb_only.py'
model = dict(
    test_cfg=dict(
        nms=dict(type='nms', iou_threshold=0.60)
    )
)
"""
    with open('configs/person2/exp_rgb_opt3.py', 'w') as f:
        f.write(config_opt3_content)
        
    config_opt3 = 'configs/person2/exp_rgb_opt3.py'
    val_res_opt3 = run_evaluation(config_opt3, checkpoint_path, split='val')
    test_res_opt3 = run_evaluation(config_opt3, checkpoint_path, split='test')
    
    # Load previous results
    metrics_file = 'results/person2_metrics.json'
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            all_results = json.load(f)
    else:
        all_results = {}
        
    # Add new results
    all_results['exp_rgb_opt1'] = {
        'val': val_res_opt1,
        'test': test_res_opt1,
        'params': '36.9 M',
        'flops': '294.18 GFLOPs', # FLOPs scale quadratically with resolution: 130.75 * (960*768)/(640*512) = 294.18 GFLOPs
        'model_size': '293.2 MB'
    }
    all_results['exp_rgb_opt2'] = {
        'val': val_res_opt2,
        'test': test_res_opt2,
        'params': '36.9 M',
        'flops': '130.75 GFLOPs',
        'model_size': '293.2 MB'
    }
    all_results['exp_rgb_opt3'] = {
        'val': val_res_opt3,
        'test': test_res_opt3,
        'params': '36.9 M',
        'flops': '130.75 GFLOPs',
        'model_size': '293.2 MB'
    }
    
    # Save all results
    with open(metrics_file, 'w') as f:
        json.dump(all_results, f, indent=4)
        
    print("Optimization sweep completed successfully.")
