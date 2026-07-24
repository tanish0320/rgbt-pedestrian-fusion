"""Single-GPU FPS/inference-time benchmark, no distributed launch required.
Usage: python tools/benchmark_simple.py <config> <checkpoint> [--n 100] [--warmup 10]
"""
import argparse
import time
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--n', type=int, default=100)
    parser.add_argument('--warmup', type=int, default=10)
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset, samples_per_gpu=1, workers_per_gpu=0, dist=False, shuffle=False)

    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model = MMDataParallel(model, device_ids=[0])
    model.eval()

    times = []
    n_total = min(args.n + args.warmup, len(dataset))
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            if i >= n_total:
                break
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(return_loss=False, rescale=True, **data)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            if i >= args.warmup:
                times.append(t1 - t0)

    avg_s = sum(times) / len(times)
    fps = 1.0 / avg_s
    print(f'RESULT n={len(times)} avg_inference_ms={avg_s*1000:.2f} fps={fps:.2f}')


if __name__ == '__main__':
    main()
