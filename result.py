"""自动加载 model 中数字版本号最高的分割模型，导出比赛 COCO RLE CSV。"""
import argparse
import csv
import json
import re
from pathlib import Path

import torch

from models import build_model
from pipeline import ROOT, instance_rles, predict_image, read_gray, rle_rows


def latest_checkpoint(model_dir):
    candidates = []
    for path in Path(model_dir).glob('*.pt'):
        match = re.fullmatch(r'(.+)_v(\d+)\.pt', path.name)
        if match:
            candidates.append((int(match.group(2)), path))
    if not candidates:
        raise FileNotFoundError(f'{model_dir} 中没有 <model>_v<版本>.pt；请先运行 train.py')
    version = max(v for v, _ in candidates)
    paths = [p for v, p in candidates if v == version]
    if len(paths) != 1:
        raise ValueError(f'最高版本 {version} 存在多个文件，请使用 --checkpoint 指定')
    return paths[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, default=ROOT / 'model')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--test-dir', type=Path, default=ROOT / 'data/test/test_images')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'csv')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--threshold', type=float, help='默认沿用模型训练配置')
    parser.add_argument('--min-area', type=int, help='默认沿用模型训练配置')
    args = parser.parse_args()
    if args.batch_size <= 0 or args.threads <= 0:
        parser.error('batch-size 和 threads 必须为正整数')
    torch.set_num_threads(args.threads)
    path = args.checkpoint or latest_checkpoint(args.model_dir)
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint.get('task') != 'segmentation' or checkpoint.get('format_version') != 1:
        raise ValueError('不是本项目的分割 checkpoint')
    config = checkpoint['config']
    threshold = config['threshold'] if args.threshold is None else args.threshold
    min_area = config['min_area'] if args.min_area is None else args.min_area
    if not 0 < threshold < 1 or min_area < 1:
        parser.error('threshold 必须在 (0,1)，min-area 必须为正整数')
    model = build_model(checkpoint['model_name']).to(args.device)
    model.load_state_dict(checkpoint['state_dict'])
    files = sorted(p for p in args.test_dir.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.tif', '.tiff'})
    if not files:
        raise FileNotFoundError(f'没有测试图片: {args.test_dir}')
    if len({p.stem for p in files}) != len(files):
        raise ValueError('测试图片存在同名 stem，无法生成唯一 filament_id')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f'submission_{checkpoint["model_name"]}_v{checkpoint["version"]:04d}.csv'
    temporary = output.with_suffix('.csv.tmp')
    empty, row_count = [], 0
    try:
        with temporary.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=['filament_id', 'segmentation_rle'])
            writer.writeheader()
            for index, image_path in enumerate(files, 1):
                image = read_gray(image_path)
                if image.shape != (2048, 2048):
                    raise ValueError(f'比赛要求 2048×2048，得到 {image.shape}: {image_path}')
                probability = predict_image(model, image, args.device, config['tile_size'], args.batch_size)
                rows = rle_rows(image_path.stem, instance_rles(probability, threshold, min_area))
                writer.writerows(rows)
                row_count += len(rows)
                if not rows:
                    empty.append(image_path.name)
                print(f'{index}/{len(files)} {image_path.name}: {len(rows)} filaments', flush=True)
        if row_count == 0:
            raise ValueError('所有测试图片都没有预测到目标；不生成可提交 CSV，请检查模型和阈值')
        temporary.replace(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    output.with_suffix('.json').write_text(json.dumps(dict(checkpoint=str(path.resolve()),
        images=len(files), instances=row_count, images_without_predictions=empty,
        threshold=threshold, min_area=min_area), indent=2), encoding='utf-8')
    print(f'CSV: {output}; {len(empty)} 张图无预测目标（不伪造占位实例）。')


if __name__ == '__main__':
    main()
