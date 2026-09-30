"""共享数据、切块、预测拼接工具。所有路径默认相对于项目目录。"""
import json
import math
import random
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
from pycocotools import mask as mask_util
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parent


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(processed):
    """读取 processed.json。image_id 为「标注者-图像名」，文件名和 2048 尺寸由此还原。"""
    annotations = json.loads(Path(processed).read_text(encoding='utf-8'))
    groups = defaultdict(list)
    images = []
    seen = set()
    for annotation in annotations:
        image_id = annotation['image_id']
        groups[image_id].append(annotation)
        if image_id in seen:
            continue
        seen.add(image_id)
        if not isinstance(image_id, str) or '-' not in image_id:
            raise ValueError(f'image_id 必须是 标注者-图像名: {image_id}')
        stem = image_id.split('-', 1)[1]
        images.append(dict(image_id=image_id, original_id=image_id,
                           file_name=f'{stem}.jpeg', height=2048, width=2048))
    return images, groups


def split_images(images, fraction=0.2, seed=42, max_images=0):
    if not 0 < fraction < 1:
        raise ValueError('val_fraction 必须在 0 与 1 之间')
    names = sorted({i['file_name'] for i in images})
    random.Random(seed).shuffle(names)
    if max_images:
        names = names[:max_images]
    if len(names) < 2:
        raise ValueError('至少需要两张不同原图才能划分训练集和验证集')
    count = max(1, min(len(names) - 1, round(len(names) * fraction)))
    val, train = set(names[:count]), set(names[count:])
    return ([i for i in images if i['file_name'] in train],
            [i for i in images if i['file_name'] in val])


def annotation_rle(annotation, height, width):
    return mask_util.merge(mask_util.frPyObjects(annotation['segmentation'], height, width))


def make_mask(record, annotations, target='segmentation'):
    height, width = record['height'], record['width']
    mask = np.zeros((height, width), dtype=np.uint8)
    for annotation in annotations:
        if target == 'segmentation':
            mask |= mask_util.decode(annotation_rle(annotation, height, width))
        elif target == 'bbox':
            x, y, w, h = annotation['bbox']
            x0, y0 = max(0, math.floor(x)), max(0, math.floor(y))
            x1, y1 = min(width, math.ceil(x + w)), min(height, math.ceil(y + h))
            mask[y0:y1, x0:x1] = 1
        else:
            raise ValueError(target)
    return mask


def read_gray(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f'不能读取图像: {path}')
    return image


class TileDataset(Dataset):
    def __init__(self, records, annotations, image_dir, tile_size=512, target='segmentation', augment=False):
        if tile_size < 16 or tile_size % 8:
            raise ValueError('tile_size 必须至少为 16 且为 8 的倍数')
        self.records, self.annotations = records, annotations
        self.image_dir, self.tile_size = Path(image_dir), tile_size
        self.target, self.augment = target, augment
        self.tiles = [(i, y, x) for i, r in enumerate(records)
                      for y in range(0, r['height'], tile_size)
                      for x in range(0, r['width'], tile_size)]

    @lru_cache(maxsize=4)
    def full_image(self, index):
        r = self.records[index]
        image = read_gray(self.image_dir / r['file_name'])
        if image.shape != (r['height'], r['width']):
            raise ValueError(f'图像尺寸和标注不一致: {r["file_name"]}')
        return image, make_mask(r, self.annotations[r['image_id']], self.target)

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, index):
        i, y, x = self.tiles[index]
        image, mask = self.full_image(i)
        size = self.tile_size
        h, w = image[y:y + size, x:x + size].shape
        inputs = np.zeros((size, size), np.float32)
        targets, valid = np.zeros_like(inputs), np.zeros_like(inputs)
        inputs[:h, :w] = image[y:y + h, x:x + w] / 255.0
        targets[:h, :w] = mask[y:y + h, x:x + w]
        valid[:h, :w] = 1
        if self.augment:
            for axis in (0, 1):
                if random.random() < 0.5:
                    inputs, targets, valid = [np.flip(a, axis=axis) for a in (inputs, targets, valid)]
        return tuple(torch.from_numpy(a.copy()).unsqueeze(0) for a in (inputs, targets, valid))


def segmentation_loss(logits, target, valid):
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, target, reduction='none')
    bce = (bce * valid).sum((1, 2, 3)) / valid.sum((1, 2, 3)).clamp_min(1)
    probabilities = logits.sigmoid() * valid
    truth = target * valid
    dice = (2 * (probabilities * truth).sum((1, 2, 3)) + 1) / (
        probabilities.sum((1, 2, 3)) + truth.sum((1, 2, 3)) + 1)
    return (bce + 1 - dice).mean()


@torch.inference_mode()
def predict_image(model, image, device, tile_size=512, batch_size=4):
    """按坐标还原概率图，不按 tile 单独提取实例。"""
    model.eval()
    h, w = image.shape
    coordinates = [(y, x) for y in range(0, h, tile_size) for x in range(0, w, tile_size)]
    probability = np.zeros((h, w), np.float32)
    for start in range(0, len(coordinates), batch_size):
        batch_coords = coordinates[start:start + batch_size]
        patches = np.zeros((len(batch_coords), 1, tile_size, tile_size), np.float32)
        for i, (y, x) in enumerate(batch_coords):
            crop = image[y:y + tile_size, x:x + tile_size]
            patches[i, 0, :crop.shape[0], :crop.shape[1]] = crop / 255.0
        predictions = model(torch.from_numpy(patches).to(device)).sigmoid().cpu().numpy()[:, 0]
        for patch, (y, x) in zip(predictions, batch_coords):
            ch, cw = min(tile_size, h - y), min(tile_size, w - x)
            probability[y:y + ch, x:x + cw] = patch[:ch, :cw]
    return probability


def instance_rles(probability, threshold=0.5, min_area=16):
    """全图八连通域：边界两侧接触（含对角）的像素属于同一实例。"""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (probability >= threshold).astype(np.uint8), connectivity=8)
    rles = []
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] >= min_area:
            rles.append(mask_util.encode(np.asfortranarray(labels == index, dtype=np.uint8)))
    return rles


def rle_rows(stem, rles):
    return [dict(filament_id=f'{stem}_{i}', segmentation_rle=rle['counts'].decode('ascii'))
            for i, rle in enumerate(rles, start=1)]
