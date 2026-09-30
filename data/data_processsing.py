"""从 COCO 标注抽出实例列表，保留原始 image_id。"""
import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = ROOT / 'train' / 'MAGFiLO_1.0_Annotations_kaggle2026_train.json'
DEFAULT_OUTPUT = ROOT / 'train' / 'processed.json'


def preprocess(source=DEFAULT_SOURCE, output=DEFAULT_OUTPUT):
    source, output = Path(source), Path(output)
    raw = json.loads(source.read_text(encoding='utf-8'))
    known = set()
    for entry in raw['images']:
        image_id = entry['id']
        if image_id in known:
            raise ValueError(f'重复的 image_id: {image_id}')
        known.add(image_id)
    annotations = []
    for entry in raw['annotations']:
        image_id = entry['image_id']
        if image_id not in known:
            raise ValueError(f'未知 image_id: {image_id}')
        polygons = entry['segmentation']
        if not isinstance(polygons, list) or not polygons:
            raise ValueError('预处理要求非空 COCO polygon 列表')
        for polygon in polygons:
            if len(polygon) < 6 or len(polygon) % 2 or not all(math.isfinite(v) for v in polygon):
                raise ValueError('无效的 polygon 坐标')
        bbox = entry['bbox']
        if len(bbox) != 4 or not all(math.isfinite(v) for v in bbox) or min(bbox[2:]) <= 0:
            raise ValueError(f'无效 bbox: {bbox}')
        annotations.append(dict(image_id=image_id, segmentation=polygons, bbox=bbox))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(annotations, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
    print(f'{len(known)} 个标注者/图像记录，{len(annotations)} 个目标 -> {output}')
    return annotations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    preprocess(args.source, args.output)


if __name__ == '__main__':
    main()
