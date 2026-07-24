"""Regenerates explorer_manifest.json and hero_pair.json from VTUAV_subset.
Run from anywhere: python reports/explorer/build_manifest.py
Requires: pip install Pillow
"""
import json, os, base64, random
from io import BytesIO
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..', '..', 'data', 'VTUAV_subset')
GRID_W = 480    # thumbnail width for the explorer grid
HERO_W = 1000   # width for the hero crossfade pair

random.seed(7)


def load_split(split):
    d = json.load(open(f'{ROOT}/annotations/{split}.json'))
    imgs = {im['id']: im for im in d['images']}
    anns_by_img = {}
    for a in d['annotations']:
        anns_by_img.setdefault(a['image_id'], []).append(a)
    return imgs, anns_by_img


def size_bucket(area):
    if area < 1024: return 'S'
    if area < 9216: return 'M'
    return 'L'


def encode_img(path, target_w, quality):
    im = Image.open(path).convert('RGB')
    w, h = im.size
    scale = target_w / w
    new_h = int(h * scale)
    im = im.resize((target_w, new_h), Image.LANCZOS)
    buf = BytesIO()
    im.save(buf, format='JPEG', quality=quality, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode('ascii')
    return f'data:image/jpeg;base64,{b64}', scale, new_h


def build_pair(split, fn, iid, anns, target_w, quality):
    rgb_path = f'{ROOT}/VTUAV_co/{split}/images/{fn}'
    ir_path = f'{ROOT}/VTUAV_ir/{split}/images/{fn}'
    if not (os.path.exists(rgb_path) and os.path.exists(ir_path)):
        return None
    rgb_b64, scale, out_h = encode_img(rgb_path, target_w, quality)
    ir_b64, _, _ = encode_img(ir_path, target_w, quality)
    boxes = []
    for a in anns:
        x, y, w, h = a['bbox']
        if w <= 0 or h <= 0:
            continue
        boxes.append({
            'x': round(x * scale, 1), 'y': round(y * scale, 1),
            'w': round(w * scale, 1), 'h': round(h * scale, 1),
            'area': a['area'], 'bucket': size_bucket(a['area'])
        })
    return {
        'split': split, 'file': fn, 'w': target_w, 'h': out_h,
        'rgb': rgb_b64, 'ir': ir_b64, 'boxes': boxes, 'n_ped': len(boxes),
    }


def build_grid_manifest():
    samples = []
    picks = {
        'train': ['00007.jpg', '01085.jpg', '01105.jpg', '03250.jpg', '06925.jpg', '00879.jpg'],
        'val': ['02283.jpg'],
        'test': [],
    }
    for split in ['train', 'val', 'test']:
        imgs, anns_by_img = load_split(split)
        fname_to_id = {im['file_name']: iid for iid, im in imgs.items()}
        chosen_ids = []
        for fn in picks.get(split, []):
            if fn in fname_to_id:
                chosen_ids.append(fname_to_id[fn])
        all_ids = [iid for iid in imgs if iid in anns_by_img]
        all_ids_sorted = sorted(all_ids, key=lambda i: len(anns_by_img[i]))
        n_target = 8 if split == 'train' else 5
        step = max(1, len(all_ids_sorted) // n_target)
        for i in range(0, len(all_ids_sorted), step):
            iid = all_ids_sorted[i]
            if iid not in chosen_ids:
                chosen_ids.append(iid)
            if len(chosen_ids) >= n_target + len(picks.get(split, [])):
                break

        for iid in chosen_ids:
            fn = imgs[iid]['file_name']
            pair = build_pair(split, fn, iid, anns_by_img.get(iid, []), GRID_W, quality=72)
            if pair:
                samples.append(pair)
                print(split, fn, 'boxes:', pair['n_ped'])

    out_path = os.path.join(HERE, 'explorer_manifest.json')
    with open(out_path, 'w') as f:
        json.dump(samples, f)
    print('grid samples:', len(samples), '| size MB:', round(os.path.getsize(out_path) / 1e6, 2))


def build_hero_pair(split='train', fn='06925.jpg'):
    imgs, anns_by_img = load_split(split)
    fname_to_id = {im['file_name']: iid for iid, im in imgs.items()}
    iid = fname_to_id[fn]
    pair = build_pair(split, fn, iid, anns_by_img.get(iid, []), HERO_W, quality=80)
    out_path = os.path.join(HERE, 'hero_pair.json')
    with open(out_path, 'w') as f:
        json.dump(pair, f)
    print('hero boxes:', pair['n_ped'], '| size KB:', round(os.path.getsize(out_path) / 1024, 1))


def build_stats_manifest():
    """Aggregate stats used by the Statistics tab — S/M/L counts, density,
    and a side-length histogram per split. All derived directly from the
    annotation JSON, matching reports/phase1_dataset_analysis.md."""
    import math
    HIST_BIN_PX = 20
    HIST_BINS = 13  # 0-260px in 20px steps, last bin is an overflow bucket

    splits = {}
    for split in ['train', 'val', 'test']:
        d = json.load(open(f'{ROOT}/annotations/{split}.json'))
        anns = d['annotations']
        n_img = len(d['images'])
        n_ann = len(anns)
        s = m = l = 0
        hist = [0] * HIST_BINS
        for a in anns:
            area = a['area']
            if area < 1024: s += 1
            elif area < 9216: m += 1
            else: l += 1
            side = math.sqrt(area)
            idx = min(int(side // HIST_BIN_PX), HIST_BINS - 1)
            hist[idx] += 1
        splits[split] = {
            'n_img': n_img, 'n_ann': n_ann,
            'mean_ped': round(n_ann / n_img, 2),
            'S': s, 'M': m, 'L': l,
            'S_pct': round(100 * s / n_ann, 2),
            'M_pct': round(100 * m / n_ann, 2),
            'L_pct': round(100 * l / n_ann, 2),
            'hist': hist,
        }

    out = {
        'bin_px': HIST_BIN_PX,
        'n_bins': HIST_BINS,
        'splits': splits,
        'pooled': {
            'n_ann': sum(v['n_ann'] for v in splits.values()),
            'S': sum(v['S'] for v in splits.values()),
            'M': sum(v['M'] for v in splits.values()),
            'L': sum(v['L'] for v in splits.values()),
        },
    }
    out_path = os.path.join(HERE, 'stats_manifest.json')
    with open(out_path, 'w') as f:
        json.dump(out, f)
    print('stats manifest written | size KB:', round(os.path.getsize(out_path) / 1024, 1))
    return out


def build_daynight_manifest():
    """Every-100th-image sampling of the train split, classified by direct visual
    inspection (see reports/phase1_technical_report.md §7) — not inferred from
    intensity alone, since 01071.jpg/05633.jpg prove mean intensity is ambiguous
    between overcast daylight and artificially-lit night scenes."""
    # (filename, mean_intensity, visual classification) — classification is the
    # result of viewing each frame directly, not computed.
    samples_meta = [
        ('00007.jpg', 121.37, 'day'),
        ('01071.jpg', 71.33, 'day'),
        ('01941.jpg', 123.06, 'day'),
        ('02980.jpg', 118.67, 'day'),
        ('03938.jpg', 118.00, 'day'),
        ('04713.jpg', 124.00, 'day'),
        ('05633.jpg', 75.02, 'night'),
        ('06791.jpg', 106.22, 'day'),
        ('07617.jpg', 129.23, 'day'),
        ('08699.jpg', 75.96, 'night'),
        ('09555.jpg', 27.22, 'night'),
        ('10511.jpg', 67.68, 'night'),
    ]
    imgs, anns_by_img = load_split('train')
    fname_to_id = {im['file_name']: iid for iid, im in imgs.items()}

    out = []
    for fn, mean_i, cls in samples_meta:
        iid = fname_to_id[fn]
        rgb_path = f'{ROOT}/VTUAV_co/train/images/{fn}'
        rgb_b64, _, _ = encode_img(rgb_path, 300, quality=70)
        out.append({
            'file': fn, 'mean_intensity': mean_i, 'classification': cls, 'rgb': rgb_b64,
        })
    out_path = os.path.join(HERE, 'daynight_manifest.json')
    with open(out_path, 'w') as f:
        json.dump(out, f)
    n_day = sum(1 for s in out if s['classification'] == 'day')
    n_night = len(out) - n_day
    print(f'day/night manifest: {n_day} day / {n_night} night | size KB:', round(os.path.getsize(out_path) / 1024, 1))


if __name__ == '__main__':
    build_grid_manifest()
    build_hero_pair()
    build_stats_manifest()
    build_daynight_manifest()
