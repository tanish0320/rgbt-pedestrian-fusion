"""Phase 3 smoke test: a few real forward+backward steps on the P2 fusion config,
to measure peak VRAM and confirm the pipeline runs end-to-end before a full training run.
Not a training script — no checkpoint saving, no eval, no LR schedule.

Usage: python tools/smoke_test_p2.py [--steps 5]
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
    parser.add_argument('--config', default='qfdet_configs/qfdet_r50_fpn_p2_vtuav.py')
    parser.add_argument('--checkpoint', default='../../checkpoints/qfdet_vtuav_pretrained.pth')
    parser.add_argument('--steps', type=int, default=5)
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    dataset = build_dataset(cfg.data.train)
    print('train dataset size:', len(dataset))
    loader = build_dataloader(
        dataset, samples_per_gpu=cfg.data.samples_per_gpu,
        workers_per_gpu=0, num_gpus=1, dist=False, shuffle=True, seed=0)

    model = build_detector(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.CLASSES = dataset.CLASSES
    model = MMDataParallel(model.cuda(), device_ids=[0])
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.optimizer['lr'],
                                 momentum=cfg.optimizer['momentum'],
                                 weight_decay=cfg.optimizer['weight_decay'])

    torch.cuda.reset_peak_memory_stats()
    it = iter(loader)
    for step in range(args.steps):
        data = next(it)
        t0 = time.time()
        optimizer.zero_grad()
        losses = model.train_step(data, optimizer)
        loss = losses['loss']
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        peak_mb = torch.cuda.max_memory_allocated() / 1e6
        print(f'step {step}: loss={loss.item():.4f} time={time.time()-t0:.2f}s peak_vram_mb={peak_mb:.0f}')

    print('SMOKE TEST PASSED')


if __name__ == '__main__':
    main()
