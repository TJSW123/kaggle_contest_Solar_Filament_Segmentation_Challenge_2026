"""把分块预测合并为概率图，并转换成比赛要求的实例 RLE。"""
import cv2
import numpy as np
import torch
from pycocotools import mask as mask_util


@torch.inference_mode()
def predict_probability(model, image, device, tile_size=512, batch_size=4):
    """分块推理并按原坐标拼回一张完整概率图。"""
    model.eval()
    height, width = image.shape
    coordinates = [
        (y, x)
        for y in range(0, height, tile_size)
        for x in range(0, width, tile_size)
    ]
    probability = np.zeros((height, width), np.float32)
    for start in range(0, len(coordinates), batch_size):
        batch_coordinates = coordinates[start:start + batch_size]
        patches = np.zeros(
            (len(batch_coordinates), 1, tile_size, tile_size),
            np.float32,
        )
        for index, (y, x) in enumerate(batch_coordinates):
            crop = image[y:y + tile_size, x:x + tile_size]
            patches[index, 0, :crop.shape[0], :crop.shape[1]] = crop / 255.0
        predictions = (
            model(torch.from_numpy(patches).to(device))
            .sigmoid()
            .cpu()
            .numpy()[:, 0]
        )
        for patch, (y, x) in zip(predictions, batch_coordinates):
            crop_height = min(tile_size, height - y)
            crop_width = min(tile_size, width - x)
            probability[y:y + crop_height, x:x + crop_width] = (
                patch[:crop_height, :crop_width]
            )
    return probability


def probability_to_rles(probability, threshold=0.5, min_area=16):
    """阈值化概率图，再以八连通域提取并过滤实例。"""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (probability >= threshold).astype(np.uint8),
        connectivity=8,
    )
    rles = []
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] >= min_area:
            mask = np.asfortranarray(labels == index, dtype=np.uint8)
            rles.append(mask_util.encode(mask))
    return rles


def rle_rows(stem, rles):
    return [
        dict(
            filament_id=f'{stem}_{index}',
            segmentation_rle=rle['counts'].decode('ascii'),
        )
        for index, rle in enumerate(rles, start=1)
    ]
