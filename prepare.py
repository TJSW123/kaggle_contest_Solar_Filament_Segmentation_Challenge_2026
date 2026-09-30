"""训练前的数据读取、划分、切块和 DataLoader 构建。"""
import json
import os
import random
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
from pycocotools import mask as mask_util
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parent
MASK_CACHE_DIR = ROOT / 'data/train/mask_cache'
NUM_WORKERS = min(8, max(0, (os.cpu_count() or 2) - 2))


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def load_data(processed):
    """读取精简标注，并还原每个标注者对应的原图记录。"""
    annotations = json.loads(Path(processed).read_text(encoding='utf-8'))
    groups = defaultdict(list)
    images = []
    seen = set()
    for annotation in annotations:
        image_id = annotation['image_id']
        groups[image_id].append(annotation)
        if image_id not in seen:
            seen.add(image_id)
            stem = image_id.split('-', 1)[1]
            images.append(dict(
                image_id=image_id,
                original_id=image_id,
                file_name=f'{stem}.jpeg',
                height=2048,
                width=2048,
            ))
    return images, groups


def split_images(images, fraction=0.2, seed=42):
    """按原图文件名划分，确保同一图片的不同标注者不会跨集合。"""
    names = sorted({image['file_name'] for image in images})
    random.Random(seed).shuffle(names)
    count = max(1, min(len(names) - 1, round(len(names) * fraction)))
    val_names = set(names[:count])
    train_names = set(names[count:])
    return (
        [image for image in images if image['file_name'] in train_names],
        [image for image in images if image['file_name'] in val_names],
    )


def annotation_rle(annotation, height, width):
    return mask_util.merge(mask_util.frPyObjects(annotation['segmentation'], height, width))


def make_mask(record, annotations):
    height, width = record['height'], record['width']
    mask = np.zeros((height, width), dtype=np.uint8)
    for annotation in annotations:
        mask |= mask_util.decode(annotation_rle(annotation, height, width))
    return mask


def _mask_cache_path(image_id):
    return MASK_CACHE_DIR / f'{image_id}.npy'


def _build_one_mask(item):
    image_id, height, width, annotations = item
    path = _mask_cache_path(image_id)
    if path.is_file():
        return image_id
    record = dict(image_id=image_id, height=height, width=width)
    np.save(path, make_mask(record, annotations))
    return image_id


def ensure_mask_cache(records, annotations):
    """把多边形标注预渲染成 uint8 mask，避免训练时反复解码。"""
    MASK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pending = [
        (record['image_id'], record['height'], record['width'], annotations[record['image_id']])
        for record in records
        if not _mask_cache_path(record['image_id']).is_file()
    ]
    if not pending:
        return
    workers = min(NUM_WORKERS or 4, len(pending))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        list(tqdm(
            pool.map(_build_one_mask, pending, chunksize=4),
            total=len(pending),
            desc='Cache masks',
            dynamic_ncols=True,
        ))


def read_gray(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f'不能读取图像: {path}')
    return image


class RecordShuffleSampler(Sampler):
    """先打乱原图，再依次输出该图全部切块，避免 mask/图像缓存失效。"""

    def __init__(self, n_records, tiles_per_image, seed=42, shuffle=True):
        self.n_records = n_records
        self.tiles_per_image = tiles_per_image
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0

    def __len__(self):
        return self.n_records * self.tiles_per_image

    def __iter__(self):
        order = list(range(self.n_records))
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(order)
        self.epoch += 1
        for record_index in order:
            start = record_index * self.tiles_per_image
            yield from range(start, start + self.tiles_per_image)


class TileDataset(Dataset):
    """把每个标注者的完整原图及 mask 按固定尺寸切块。"""

    def __init__(self, records, annotations, image_dir, tile_size=512, augment=False):
        self.records = records
        self.annotations = annotations
        self.image_dir = Path(image_dir)
        self.tile_size = tile_size
        self.augment = augment
        self.tiles_per_image = (records[0]['height'] // tile_size) * (records[0]['width'] // tile_size)
        self.tiles = [
            (index, y, x)
            for index, record in enumerate(records)
            for y in range(0, record['height'], tile_size)
            for x in range(0, record['width'], tile_size)
        ]

    @lru_cache(maxsize=32)
    def full_image(self, index):
        record = self.records[index]
        image = read_gray(self.image_dir / record['file_name'])
        mask = np.load(_mask_cache_path(record['image_id']), mmap_mode='r')
        return image, mask

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, index):
        record_index, y, x = self.tiles[index]
        image, mask = self.full_image(record_index)
        size = self.tile_size
        crop = image[y:y + size, x:x + size]
        h, w = crop.shape
        inputs = np.empty((size, size), np.float32)
        targets = np.zeros((size, size), np.float32)
        valid = np.zeros((size, size), np.float32)
        if h == size and w == size:
            inputs[...] = crop
            inputs /= 255.0
            targets[...] = mask[y:y + size, x:x + size]
            valid[...] = 1
        else:
            inputs.fill(0)
            inputs[:h, :w] = crop / 255.0
            targets[:h, :w] = mask[y:y + h, x:x + w]
            valid[:h, :w] = 1
        if self.augment:
            for axis in (0, 1):
                if random.random() < 0.5:
                    inputs, targets, valid = [
                        np.flip(array, axis=axis) for array in (inputs, targets, valid)
                    ]
        return tuple(
            torch.from_numpy(np.ascontiguousarray(array)).unsqueeze(0)
            for array in (inputs, targets, valid)
        )


def build_loaders(processed, image_dir, tile_size, batch_size, val_fraction,
                  seed, device):
    """一次完成数据加载、原图划分和三个 DataLoader 的构建。"""
    images, annotations = load_data(processed)
    train_records, val_records = split_images(images, val_fraction, seed)
    image_dir = Path(image_dir)
    ensure_mask_cache(train_records + val_records, annotations)

    dataset_options = dict(
        annotations=annotations,
        image_dir=image_dir,
        tile_size=tile_size,
    )
    loader_options = dict(
        batch_size=batch_size,
        num_workers=NUM_WORKERS,
        pin_memory=str(device).startswith('cuda'),
        persistent_workers=NUM_WORKERS > 0,
    )
    if NUM_WORKERS > 0:
        loader_options['prefetch_factor'] = 4
    train_set = TileDataset(train_records, augment=True, **dataset_options)
    train_eval_set = TileDataset(train_records, **dataset_options)
    val_set = TileDataset(val_records, **dataset_options)
    training = DataLoader(
        train_set,
        sampler=RecordShuffleSampler(
            len(train_records), train_set.tiles_per_image, seed=seed, shuffle=True,
        ),
        **loader_options,
    )
    train_eval = DataLoader(
        train_eval_set,
        sampler=RecordShuffleSampler(
            len(train_records), train_eval_set.tiles_per_image, seed=seed, shuffle=False,
        ),
        **loader_options,
    )
    validation = DataLoader(
        val_set,
        sampler=RecordShuffleSampler(
            len(val_records), val_set.tiles_per_image, seed=seed, shuffle=False,
        ),
        **loader_options,
    )
    return train_records, val_records, annotations, (training, train_eval, validation)


def segmentation_loss(logits, target, valid):
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, target, reduction='none'
    )
    bce = (bce * valid).sum((1, 2, 3)) / valid.sum((1, 2, 3)).clamp_min(1)
    probabilities = logits.sigmoid() * valid
    truth = target * valid
    dice = (2 * (probabilities * truth).sum((1, 2, 3)) + 1) / (
        probabilities.sum((1, 2, 3)) + truth.sum((1, 2, 3)) + 1
    )
    return (bce + 1 - dice).mean()


def run_epoch(model, loader, device, optimizer=None, description=''):
    """训练一轮；未传 optimizer 时只计算平均 loss。"""
    model.train(optimizer is not None)
    total = 0.0
    count = 0
    batches = tqdm(
        loader, desc=description,
        leave=optimizer is not None, dynamic_ncols=True,
    )
    with torch.set_grad_enabled(optimizer is not None):
        for inputs, target, valid in batches:
            inputs, target, valid = (
                tensor.to(device, non_blocking=True)
                for tensor in (inputs, target, valid)
            )
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            loss = segmentation_loss(model(inputs), target, valid)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            total += float(loss.detach()) * len(inputs)
            count += len(inputs)
            batches.set_postfix(loss=f'{total / count:.4f}')
    return total / count
