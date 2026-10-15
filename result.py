"""自动加载 model 中数字版本号最高的分割模型，导出比赛 COCO RLE CSV。"""
import csv
import json
import re
from pathlib import Path

import torch

from models import build_model
from output import predict_probability, probability_to_rles, rle_rows
from prepare import ROOT, read_gray

MODEL_DIR = ROOT / 'model'
CHECKPOINT = None       # 例如 ROOT / 'model/resnet18/resnet18_unet_v0001.pt'；None 自动选最新版本
TEST_DIR = ROOT / 'data/test/test_images'
OUTPUT_DIR = ROOT / 'csv'
BATCH_SIZE = 4
THREADS = 4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
THRESHOLD = None       # None 表示沿用模型训练配置
MIN_AREA = None        # None 表示沿用模型训练配置


def latest_checkpoint(model_dir):
    candidates = []
    for path in Path(model_dir).rglob('*.pt'):
        match = re.fullmatch(r'(.+)_v(\d+)\.pt', path.name)
        if match:
            candidates.append((int(match.group(2)), path))
    if not candidates:
        raise FileNotFoundError(f'{model_dir} 中没有 <model>_v<版本>.pt；请先运行 train.py')
    version = max(v for v, _ in candidates)
    paths = [p for v, p in candidates if v == version]
    if len(paths) != 1:
        raise ValueError(f'最高版本 {version} 存在多个文件，请设置 CHECKPOINT')
    return paths[0]


def main():
    if BATCH_SIZE <= 0 or THREADS <= 0:
        raise ValueError('BATCH_SIZE 和 THREADS 必须为正整数')
    torch.set_num_threads(THREADS)
    path = Path(CHECKPOINT) if CHECKPOINT is not None else latest_checkpoint(MODEL_DIR)
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint.get('task') != 'segmentation' or checkpoint.get('format_version') != 1:
        raise ValueError('不是本项目的分割 checkpoint')
    config = checkpoint['config']
    threshold = config['threshold'] if THRESHOLD is None else THRESHOLD
    min_area = config['min_area'] if MIN_AREA is None else MIN_AREA
    if not 0 < threshold < 1 or min_area < 1:
        raise ValueError('THRESHOLD 必须在 (0,1)，MIN_AREA 必须为正整数')
    # checkpoint 已包含编码器权重，推理时不应再下载 ImageNet 权重。
    model = build_model(checkpoint['model_name'], pretrained=False).to(DEVICE)
    model.load_state_dict(checkpoint['state_dict'])
    files = sorted(p for p in TEST_DIR.iterdir()
                   if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.tif', '.tiff'})
    if not files:
        raise FileNotFoundError(f'没有测试图片: {TEST_DIR}')
    if len({p.stem for p in files}) != len(files):
        raise ValueError('测试图片存在同名 stem，无法生成唯一 filament_id')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_DIR / f'submission_{checkpoint["model_name"]}_v{checkpoint["version"]:04d}.csv'
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
                probability = predict_probability(
                    model, image, DEVICE, config['tile_size'], BATCH_SIZE,
                )
                rows = rle_rows(
                    image_path.stem,
                    probability_to_rles(probability, threshold, min_area),
                )
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
